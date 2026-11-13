from __future__ import annotations

import torch
import torch.nn as nn

from ..v1.modules_v1 import RMSNorm, SwiGLUFeedForward

"""
v3's two new modules: a Mixture-of-Experts feed-forward network and a
per-token depth router. Attention is v2's, unchanged.
"""


class MoEFeedForward(nn.Module):
    """
    Mixture of Experts (Shazeer et al., 2017; Switch/Mixtral-style).

    Replaces the single dense SwiGLU FFN with `num_experts` independent
    SwiGLU FFNs plus a router. For every token the router scores all
    experts, the top_k best are run, and their outputs are summed,
    weighted by the (renormalised) router probabilities:

        p = softmax(W_r x)                      one probability per expert
        S = top_k(p);  w_e = p_e / sum_{S} p    weights over the chosen few
        y = sum_{e in S} w_e * Expert_e(x)

    Total parameters grow ~num_experts-fold, but each token only pays for
    top_k experts - that gap between capacity and per-token compute is the
    whole point. Because the weights multiply the expert outputs, the
    router receives gradients and learns which expert suits which token.

    Load balancing: left alone, routers collapse onto a few favourite
    experts (they get trained most, so they get chosen more). The
    auxiliary loss from the Switch Transformer discourages it:

        L = E * sum_e f_e * P_e

    f_e is the fraction of routing slots sent to expert e (not
    differentiable) and P_e the mean router probability for e (which is).
    The product is minimised, at L = 1, when both are uniform; the
    gradient through P_e pushes probability away from overloaded experts.
    Padding tokens are excluded from both averages.

    Implementation notes: dropless - no per-expert capacity limit, no
    dropped tokens - and experts run one after another on just their
    tokens (a loop, not a fused grouped matmul), so it is correct and
    simple rather than fast on a GPU.

    Learnable parameters (no biases): router D*E; each expert 3*D*F
    total = D*E + E*3*D*F
    [D=256, F=1024, E=4: 1,024 + 3,145,728 = 3,146,752 vs. dense 786,432]
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        dropout_param: float = 0.1,
    ):
        super().__init__()

        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")

        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(d_model, num_experts, bias=False)

        self.experts = nn.ModuleList([
            SwiGLUFeedForward(d_model, d_ff, dropout_param)
            for _ in range(num_experts)
        ])

        # Set on every forward when True; diagnostics only (kept off by
        # default so the forward pass has no side effects).
        self.track_usage = False
        self.last_usage: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, S, D); token_mask: (B, S), True for real (non-padding)
        tokens. Returns (output, unweighted load-balancing loss).
        """

        batch_size, seq_len, d_model = x.shape

        flat = x.reshape(-1, d_model)

        # Routing in fp32 even under autocast: softmax over experts is
        # sensitive and the aux loss is a small number.
        probs = torch.softmax(self.router(flat).float(), dim=-1)

        top_probs, top_experts = probs.topk(self.top_k, dim=-1)
        top_weights = top_probs / top_probs.sum(dim=-1, keepdim=True)

        output = flat.new_zeros(flat.shape)

        for expert_id, expert in enumerate(self.experts):

            token_index, slot = (top_experts == expert_id).nonzero(as_tuple=True)

            if token_index.numel() == 0:
                continue

            expert_output = expert(flat[token_index])
            weight = top_weights[token_index, slot, None].to(expert_output.dtype)

            output.index_add_(0, token_index, (expert_output * weight).to(output.dtype))

        # ---- load-balancing loss over real tokens only
        if token_mask is None:
            valid = torch.ones(flat.shape[0], device=x.device)
        else:
            valid = token_mask.reshape(-1).float()

        num_valid = valid.sum().clamp(min=1.0)

        assignments = torch.zeros_like(probs).scatter(1, top_experts, 1.0)

        fraction = (assignments * valid[:, None]).sum(0) / (num_valid * self.top_k)
        mean_prob = (probs * valid[:, None]).sum(0) / num_valid

        balance_loss = self.num_experts * (fraction.detach() * mean_prob).sum()

        if self.track_usage:
            self.last_usage = fraction.detach()

        return output.view(batch_size, seq_len, d_model), balance_loss


class DepthRouter(nn.Module):
    """
    Per-token "should this token use this layer?" gate - conditional
    computation / dynamic depth, in the spirit of Mixture-of-Depths
    (Raposo et al., 2024) but with a per-token threshold instead of a
    per-sequence top-k:

        score = W x;  p = sigmoid(score);  use = p > 0.5

    A token that skips a layer simply keeps its residual stream
    unchanged, so "easy" tokens can leave the stack early and "hard" ones
    use every layer. A per-token threshold (rather than "the top 50% of
    this sequence") matters for generation: top-k over a sequence would
    need to see future tokens, but a threshold decides each token alone,
    so training and cached decoding agree.

    The hard 0/1 decision isn't differentiable, so the gate uses a
    straight-through estimator: the forward value is the hard decision,
    the backward gradient is that of the soft probability p. Nothing
    would stop the router from always saying "use it" though, so an
    auxiliary loss pulls the mean probability toward `capacity`
    (the target fraction of tokens that should use the layer):

        L = (mean_p - capacity)^2

    The bias starts positive so training begins with (nearly) every
    token using every layer, and learns to skip from there.

    Honest limitation: a skipped token's update is *masked out*, not
    avoided - attention and the FFN still run for the whole batch. This
    module demonstrates and trains the routing; turning it into real
    savings would need gathering the selected tokens into a smaller
    tensor, which interacts with attention, RoPE positions and the KV
    cache.

    Learnable parameters: D (weights) + 1 (bias) + D (RMSNorm gamma)
    """

    def __init__(
        self,
        d_model: int,
        capacity: float = 0.5,
    ):
        super().__init__()

        if not 0.0 < capacity <= 1.0:
            raise ValueError("depth_capacity must be in (0, 1]")

        self.capacity = capacity

        self.norm = RMSNorm(d_model)
        self.score = nn.Linear(d_model, 1)

        self.last_use_fraction: torch.Tensor | None = None

        self.track_usage = False

    def initialize(self) -> None:
        """Start by letting (almost) every token through (p ~ 0.88)."""

        nn.init.zeros_(self.score.weight)
        nn.init.constant_(self.score.bias, 2.0)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (gate (B, S, 1) with value 0/1 forward and soft gradient,
        unweighted budget loss).
        """

        soft = torch.sigmoid(self.score(self.norm(x)).float().squeeze(-1))

        hard = (soft > 0.5).float()

        gate = hard + soft - soft.detach()

        if token_mask is None:
            mean_soft = soft.mean()
        else:
            valid = token_mask.float()
            mean_soft = (soft * valid).sum() / valid.sum().clamp(min=1.0)

        if self.track_usage:
            self.last_use_fraction = (
                hard.mean() if token_mask is None
                else (hard * token_mask.float()).sum() / token_mask.float().sum().clamp(min=1.0)
            ).detach()

        return gate.unsqueeze(-1), (mean_soft - self.capacity) ** 2
