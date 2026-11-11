import torch
import torch.nn as nn

"""
Building blocks shared by EncoderV1 and DecoderV1. v0 didn't need a file
like this - it only used off-the-shelf nn.MultiheadAttention/nn.Sequential
blocks. v1 replaces those with custom modules because RoPE has to be
injected *inside* attention (after the Q/K projection, before the dot
product), which nn.MultiheadAttention doesn't expose a hook for.
"""


class RMSNorm(nn.Module):
    """
    RMSNorm(x) = x / sqrt(mean(x^2, dim=-1) + eps) * gamma

    Compare to LayerNorm, which v0 uses:
        LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

    RMSNorm drops the mean-subtraction and the learned bias (beta) -
    it only rescales by the root-mean-square of the activations. This
    is slightly cheaper (one fewer reduction) and uses fewer parameters,
    and in practice matches LayerNorm's quality in transformers. It's
    the normalization used by LLaMA, Mistral, Gemma, and most modern
    open LLMs.

    Learnable parameters: gamma: D   (vs. LayerNorm's gamma: D + beta: D)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        return x * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - Su et al., 2021.

    v0 encodes position by *adding* a learned vector to the token
    embedding once, up front (absolute position). RoPE instead *rotates*
    each attention head's query/key vectors by an angle proportional to
    their position, applied fresh inside every attention layer. This has
    two consequences that are why most LLMs since GPT-NeoX/LLaMA use it:

      1. Zero learned parameters - just a fixed, precomputed table of
         angles (register_buffer below), vs. v0's Smax*D learned
         position-embedding table.

      2. Relative position for free. Rotating a vector by angle m*theta
         is applying a rotation matrix R(m*theta). For two rotated
         vectors (q at position m, k at position n), their dot product
         works out to depend only on R((m-n)*theta) - i.e. only the
         *difference* m-n survives, not the absolute positions. This
         generalizes naturally to sequence lengths longer than anything
         seen in training, unlike a fixed-size absolute position table.

    Implementation: the "rotate-half" convention used by GPT-NeoX/LLaMA
    (simpler to implement than the original paper's interleaved-pairs
    version, and numerically equivalent up to a fixed permutation of the
    head dimension). Split each head_dim-sized vector in half and treat
    (x[:d/2], x[d/2:]) as the real/imaginary parts of d/2 complex
    numbers; multiply each by e^(i * position * theta_i), where theta_i
    shrinks geometrically across the pair index i (low pairs rotate
    fast - high frequency - high pairs rotate slow - low frequency;
    same spirit as the sinusoidal position encoding in the original
    "Attention Is All You Need").
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")

        # theta_i = theta ** (-2i / head_dim), for i = 0 .. head_dim/2 - 1
        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )

        positions = torch.arange(max_seq_len).float()

        # angles[m, i] = m * theta_i        shape: (max_seq_len, head_dim/2)
        angles = torch.outer(positions, inv_freq)

        # Duplicate so each half of the rotate-half split gets the same
        # angle: [theta_0 .. theta_{d/2-1}, theta_0 .. theta_{d/2-1}]
        angles = torch.cat([angles, angles], dim=-1)

        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:

    half = x.shape[-1] // 2
    first_half, second_half = x[..., :half], x[..., half:]

    return torch.cat([-second_half, first_half], dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    x: (batch, heads, seq_len, head_dim)
    cos/sin: (seq_len, head_dim) - broadcast over the batch/heads dims.
    """

    return x * cos + _rotate_half(x) * sin


class MultiHeadAttentionV1(nn.Module):
    """
    Custom multi-head attention. Replaces v0's nn.MultiheadAttention so
    RoPE can be spliced in between the Q/K projection and the scaled dot
    product - a hook the built-in module doesn't provide.

    Used for both self-attention (use_rope=True, since relative position
    matters within one sequence) and cross-attention (use_rope=False -
    see DecoderLayerV1 for why RoPE is skipped there).

    Learnable parameters: Wq, Wk, Wv, Wo: D*D each, no biases (bias=False
    is the common pairing with Pre-Norm/RMSNorm architectures - one less
    thing to learn, negligible effect on quality at these scales).
    total = 4*D*D   (vs. v0's 4*D*D + 4*D with biases)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_param: float = 0.1,
        use_rope: bool = False,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout_param)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape

        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)

        return x.transpose(1, 2)  # (B, H, S, head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, num_heads, seq_len, head_dim = x.shape

        x = x.transpose(1, 2).contiguous()

        return x.view(batch_size, seq_len, num_heads * head_dim)

    def forward(
        self,
        query_input: torch.Tensor,
        key_value_input: torch.Tensor,
        rope: RotaryEmbedding | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        q = self._split_heads(self.w_q(query_input))
        k = self._split_heads(self.w_k(key_value_input))
        v = self._split_heads(self.w_v(key_value_input))

        if self.use_rope:

            if rope is None:
                raise ValueError("use_rope=True requires a RotaryEmbedding instance")

            cos, sin = rope(q.shape[-2])
            q = apply_rotary_pos_emb(q, cos, sin)

            cos, sin = rope(k.shape[-2])
            k = apply_rotary_pos_emb(k, cos, sin)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        if key_padding_mask is not None:
            # (B, S_k) -> (B, 1, 1, S_k): broadcasts over heads and query
            # positions. True means "this key position is padding, mask
            # it out" - same convention as v0/nn.MultiheadAttention.
            mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = self._merge_heads(context)

        return self.w_o(context)


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU(x) = (Swish(x W1) * (x W3)) W2
    where Swish(z) = z * sigmoid(z)  (= nn.SiLU)

    A *gated* FFN, vs. v0's plain GELU FFN (GELU(x W1) W2). W1 produces
    a Swish-activated "signal", W3 produces a parallel, unactivated
    "gate" that elementwise-multiplies it, and W2 projects back down to
    d_model. Shazeer (2020) found gated variants like this consistently
    outperform plain activated FFNs at matched parameter/compute budgets;
    it's the FFN used by LLaMA, PaLM, Mistral, and most modern LLMs.

    Learnable parameters: W1, W3: D*F each; W2: F*D   (no biases)
    total = 3*D*F   (vs. v0's GELU FFN: 2*D*F + D+F+D with biases)
    Because of the extra matrix, real-world v1-style models often shrink
    d_ff to ~2/3 of what they'd use for a GELU FFN, to keep total FFN
    parameters/compute comparable. Left as an explicit config choice
    here rather than baked in, so it's easy to see the actual parameter
    cost of each choice.
    """

    def __init__(self, d_model: int, d_ff: int, dropout_param: float = 0.1):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout_param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        hidden = self.activation(self.w1(x)) * self.w3(x)

        return self.w2(self.dropout(hidden))
