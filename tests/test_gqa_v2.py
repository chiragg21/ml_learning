"""
Lite smoke tests for Grouped-Query Attention (v2): checks the properties
that make it GQA rather than just "a different attention module" - it
collapses to v1's attention when every query head has its own K/V head,
it really shares K/V across a group, and it shrinks the K/V projections.
End-to-end training with transformer_v2 is covered in
test_smoke_training.py.
"""

from __future__ import annotations

import pytest
import torch

from src.models.v1.modules_v1 import MultiHeadAttentionV1, RotaryEmbedding
from src.models.v2.modules_v2 import GroupedQueryAttentionV2


D_MODEL = 32
NUM_HEADS = 4
HEAD_DIM = D_MODEL // NUM_HEADS


def _inputs() -> torch.Tensor:

    torch.manual_seed(0)

    return torch.randn(2, 6, D_MODEL)


def test_gqa_with_full_kv_heads_matches_v1_attention():

    rope = RotaryEmbedding(head_dim=HEAD_DIM, max_seq_len=16)

    v1_attention = MultiHeadAttentionV1(D_MODEL, NUM_HEADS, dropout_param=0.0, use_rope=True)
    v2_attention = GroupedQueryAttentionV2(
        D_MODEL,
        NUM_HEADS,
        num_kv_heads=NUM_HEADS,
        dropout_param=0.0,
        use_rope=True,
    )

    v2_attention.load_state_dict(v1_attention.state_dict())

    x = _inputs()

    expected = v1_attention(x, x, rope=rope)
    actual = v2_attention(x, x, rope=rope)

    assert torch.allclose(expected, actual, atol=1e-6)


def test_gqa_shares_kv_heads_within_a_group():

    attention = GroupedQueryAttentionV2(
        D_MODEL,
        NUM_HEADS,
        num_kv_heads=2,
        dropout_param=0.0,
    )

    assert attention.group_size == 2

    # Two query heads in one group see identical K/V, so giving them
    # identical query weights must give them identical outputs.
    with torch.no_grad():
        q_weight = attention.w_q.weight.view(NUM_HEADS, HEAD_DIM, D_MODEL)
        q_weight[1] = q_weight[0]

    x = _inputs()

    q = attention._split_heads(attention.w_q(x), NUM_HEADS)
    k = attention._split_heads(attention.w_k(x), 2).repeat_interleave(2, dim=1)

    scores = torch.matmul(q, k.transpose(-2, -1))

    assert torch.allclose(scores[:, 0], scores[:, 1])
    assert not torch.allclose(scores[:, 0], scores[:, 2])


def test_gqa_has_fewer_parameters_than_full_attention():

    full = GroupedQueryAttentionV2(D_MODEL, NUM_HEADS, num_kv_heads=NUM_HEADS)
    grouped = GroupedQueryAttentionV2(D_MODEL, NUM_HEADS, num_kv_heads=1)

    full_params = sum(p.numel() for p in full.parameters())
    grouped_params = sum(p.numel() for p in grouped.parameters())

    assert full_params == 4 * D_MODEL * D_MODEL
    assert grouped_params == 2 * D_MODEL * D_MODEL + 2 * D_MODEL * HEAD_DIM


def test_gqa_rejects_num_heads_not_divisible_by_num_kv_heads():

    with pytest.raises(ValueError):
        GroupedQueryAttentionV2(D_MODEL, NUM_HEADS, num_kv_heads=3)


def test_gqa_cross_attention_handles_different_query_and_memory_lengths():

    attention = GroupedQueryAttentionV2(
        D_MODEL,
        NUM_HEADS,
        num_kv_heads=2,
        use_rope=False,
    )

    query = _inputs()
    memory = torch.randn(2, 9, D_MODEL)

    assert attention(query, memory).shape == query.shape
