from __future__ import annotations

import torch
import torch.nn as nn

from ..v1.modules_v1 import RotaryEmbedding, apply_rotary_pos_emb

"""
v2 only changes attention, so this file holds just that one module -
RMSNorm, RotaryEmbedding and SwiGLUFeedForward are reused unchanged
from modules_v1.
"""


class GroupedQueryAttentionV2(nn.Module):
    """
    Grouped-Query Attention (GQA) - Ainslie et al., 2023.

    v1 gives every attention head its own query, key and value
    projection. GQA keeps one query projection per head but shares each
    key/value head across a *group* of query heads:

        num_heads = 8, num_kv_heads = 2  ->  group_size = 4
        query heads 0-3 all read K/V head 0; query heads 4-7 read K/V head 1

    The two ends of the spectrum are named variants: num_kv_heads ==
    num_heads is ordinary multi-head attention (exactly v1), and
    num_kv_heads == 1 is multi-query attention. The reason to pick the
    middle is inference memory: the KV cache stores K and V for every
    past token, so its size scales with num_kv_heads, not num_heads.
    Shrinking it by group_size costs very little quality (LLaMA 2 70B,
    Mistral and most current models do this) and is what makes the KV
    cache - the next v2 item - affordable.

    Implementation: K and V are projected to num_kv_heads heads, RoPE is
    applied to them at that smaller head count, and only then are they
    expanded with repeat_interleave so each group's K/V lines up with its
    query heads. Rotating before expanding is equivalent (every copy gets
    the same rotation) but does group_size times less rotary work.

    Learnable parameters (no biases): with kv_dim = num_kv_heads*head_dim
    Wq, Wo: D*D each; Wk, Wv: D*kv_dim each
    total = 2*D*D + 2*D*kv_dim
    [D=256, H=8, num_kv_heads=2: 2*65536 + 2*256*64 = 163,840
     vs. v1's 4*D*D = 262,144]
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout_param: float = 0.1,
        use_rope: bool = False,
    ):
        super().__init__()

        if num_kv_heads is None:
            num_kv_heads = num_heads

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope

        kv_dim = num_kv_heads * self.head_dim

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, kv_dim, bias=False)
        self.w_v = nn.Linear(d_model, kv_dim, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout_param)

    def _split_heads(
        self,
        x: torch.Tensor,
        num_heads: int,
    ) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape

        x = x.view(batch_size, seq_len, num_heads, self.head_dim)

        return x.transpose(1, 2)  # (B, num_heads, S, head_dim)

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

        q = self._split_heads(self.w_q(query_input), self.num_heads)
        k = self._split_heads(self.w_k(key_value_input), self.num_kv_heads)
        v = self._split_heads(self.w_v(key_value_input), self.num_kv_heads)

        if self.use_rope:

            if rope is None:
                raise ValueError("use_rope=True requires a RotaryEmbedding instance")

            cos, sin = rope(q.shape[-2])
            q = apply_rotary_pos_emb(q, cos, sin)

            cos, sin = rope(k.shape[-2])
            k = apply_rotary_pos_emb(k, cos, sin)

        # (B, num_kv_heads, S, hd) -> (B, num_heads, S, hd); consecutive
        # query heads share a K/V head, matching the grouping above.
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = self._merge_heads(context)

        return self.w_o(context)
