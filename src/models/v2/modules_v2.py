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

    Optional local (sliding-window) attention, for self-attention only:
    with window_size = W each token attends only to positions at distance
    less than W from itself (itself plus W-1 neighbours on each side; the
    decoder's causal mask leaves just the W-1 previous ones). Stacking
    layers still lets information travel further - after L layers a
    token is influenced by tokens up to L*(W-1) positions away - which is
    how Mistral and Longformer-style models cover long inputs. Here the
    window is applied as a mask on the full score matrix, so this
    version changes *what* is attended to, not the O(n^2) cost; it is
    the semantic half of sparse attention that a banded kernel would
    later make cheap. window_size=None is full attention.

    Banded kernel (banded=True, needs window_size): computes only the
    2W-1 scores each query can actually use. For every query i it gathers
    the keys at positions i-(W-1) .. i+(W-1) (out-of-range ones masked)
    and attends over that band, so scores are (S x 2W-1) instead of
    (S x S): O(S*W) compute and memory. The result is the same as the
    masked path; tests/test_banded_attention_v2.py checks that.

    KV cache (cache=dict, self-attention): during generation the keys and
    values of tokens already produced never change, so each step projects
    only the new token and appends its (already RoPE-rotated, kv-head-
    sized) K/V to the cache instead of recomputing the whole prefix. With
    a cache, self-attention is causal by construction - positions, not a
    supplied mask, decide what is visible - and with a window the cache
    is trimmed to the W-1 keys still in reach. Cross-attention
    (cross_attention=True) caches the projected encoder memory once,
    since it is the same at every step.

    Packed sequences (segment_mask): several examples share one row;
    a (batch, query, key) boolean mask blocks attention across examples.
    RoPE needs no reset per example because it only encodes relative
    distance.

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
        window_size: int | None = None,
        banded: bool = False,
        cross_attention: bool = False,
    ):
        super().__init__()

        if num_kv_heads is None:
            num_kv_heads = num_heads

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        if window_size is not None and window_size < 1:
            raise ValueError("window_size must be at least 1")

        if banded and window_size is None:
            raise ValueError("banded attention requires window_size")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope
        self.window_size = window_size
        self.banded = banded
        self.cross_attention = cross_attention

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

    # ==================================================
    # Local Attention
    # ==================================================

    def _local_window_mask(
        self,
        query_len: int,
        key_len: int,
        device: torch.device,
        query_start: int = 0,
        key_start: int = 0,
    ) -> torch.Tensor:

        if query_len != key_len and query_start == key_start:
            raise ValueError("window_size is only valid for self-attention")

        query_positions = query_start + torch.arange(query_len, device=device)
        key_positions = key_start + torch.arange(key_len, device=device)
        distance = query_positions[:, None] - key_positions[None, :]

        # abs() so one rule serves both sides: the encoder sees a band
        # around each token, and the decoder's causal mask already removes
        # the future half of that band.
        return distance.abs() >= self.window_size

    def _banded_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        segment_mask: torch.Tensor | None,
    ) -> torch.Tensor:

        batch_size, _, seq_len, head_dim = q.shape

        reach = self.window_size - 1

        # key_index[i, t] = i - reach + t: the key at band slot t of query i
        offsets = torch.arange(-reach, reach + 1, device=q.device)
        key_index = torch.arange(seq_len, device=q.device)[:, None] + offsets[None, :]

        out_of_range = (key_index < 0) | (key_index >= seq_len)
        key_index = key_index.clamp(0, seq_len - 1)

        # (B, kv_heads, S, band, hd); queries are grouped so each group
        # reads its shared K/V head without repeat_interleave copies.
        k_band = k[:, :, key_index]
        v_band = v[:, :, key_index]

        q = q.reshape(
            batch_size,
            self.num_kv_heads,
            self.group_size,
            seq_len,
            head_dim,
        )

        scores = torch.einsum("bkgid,bkitd->bkgit", q, k_band)
        scores = scores * self.head_dim ** -0.5

        masked = out_of_range[None, None, None]

        if attn_mask is not None:
            rows = torch.arange(seq_len, device=q.device)[:, None]
            masked = masked | attn_mask[rows, key_index][None, None, None]

        if segment_mask is not None:
            rows = torch.arange(seq_len, device=q.device)[:, None]
            masked = masked | segment_mask[:, rows, key_index][:, None, None]

        if key_padding_mask is not None:
            masked = masked | key_padding_mask[:, key_index][:, None, None]

        scores = scores.masked_fill(masked, float("-inf"))

        attn_weights = torch.nan_to_num(torch.softmax(scores, dim=-1))
        attn_weights = self.dropout(attn_weights)

        context = torch.einsum("bkgit,bkitd->bkgid", attn_weights, v_band)

        return context.reshape(batch_size, self.num_heads, seq_len, head_dim)

    # ==================================================
    # Forward
    # ==================================================

    def forward(
        self,
        query_input: torch.Tensor,
        key_value_input: torch.Tensor,
        rope: RotaryEmbedding | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        segment_mask: torch.Tensor | None = None,
        cache: dict | None = None,
    ) -> torch.Tensor:

        use_cache = cache is not None and not self.cross_attention

        # Tokens already in the cache (a trimmed window cache still
        # remembers how many it has seen, so positions stay correct).
        past_len = cache.get("seen", 0) if use_cache else 0

        q = self._split_heads(self.w_q(query_input), self.num_heads)

        if (
            cache is not None
            and self.cross_attention
            and "k" in cache
        ):
            k, v = cache["k"], cache["v"]
        else:
            k = self._split_heads(self.w_k(key_value_input), self.num_kv_heads)
            v = self._split_heads(self.w_v(key_value_input), self.num_kv_heads)

        if self.use_rope:

            if rope is None:
                raise ValueError("use_rope=True requires a RotaryEmbedding instance")

            cos, sin = rope(past_len + q.shape[-2])
            q = apply_rotary_pos_emb(q, cos[past_len:], sin[past_len:])

            cos, sin = rope(past_len + k.shape[-2])
            k = apply_rotary_pos_emb(k, cos[past_len:], sin[past_len:])

        query_len = q.shape[-2]
        key_start = 0

        if use_cache:

            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=-2)
                v = torch.cat([cache["v"], v], dim=-2)

            seen = past_len + query_len

            if self.window_size is not None:
                keep = self.window_size - 1 + query_len
                k, v = k[..., -keep:, :], v[..., -keep:, :]

            cache["k"], cache["v"], cache["seen"] = k, v, seen
            key_start = seen - k.shape[-2]

        elif cache is not None:
            cache["k"], cache["v"] = k, v

        if (
            self.banded
            and not use_cache
            and q.shape[-2] == k.shape[-2]
        ):
            context = self._banded_attention(
                q,
                k,
                v,
                attn_mask,
                key_padding_mask,
                segment_mask,
            )

            return self.w_o(self._merge_heads(context))

        # (B, num_kv_heads, S, hd) -> (B, num_heads, S, hd); consecutive
        # query heads share a K/V head, matching the grouping above.
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if use_cache:

            # Visible = not in the future of the query. Nothing else
            # supplies causality here, because the decoder only passes
            # the new tokens.
            query_positions = past_len + torch.arange(query_len, device=scores.device)
            key_positions = key_start + torch.arange(k.shape[-2], device=scores.device)

            scores = scores.masked_fill(
                key_positions[None, :] > query_positions[:, None],
                float("-inf"),
            )

        if self.window_size is not None:
            window_mask = self._local_window_mask(
                q.shape[-2],
                k.shape[-2],
                device=scores.device,
                query_start=past_len,
                key_start=key_start,
            )
            scores = scores.masked_fill(window_mask, float("-inf"))

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        if segment_mask is not None:
            scores = scores.masked_fill(segment_mask[:, None], float("-inf"))

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)

        if self.window_size is not None or segment_mask is not None:
            # A <PAD> query whose whole window (or whole packed segment)
            # is also <PAD> has every key masked, and softmax over all
            # -inf is NaN. Full attention can't hit this (some real token
            # is always in reach), but NaN at a padding position still
            # poisons the loss and gradients, so those rows become zeros
            # instead.
            attn_weights = torch.nan_to_num(attn_weights)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = self._merge_heads(context)

        return self.w_o(context)
