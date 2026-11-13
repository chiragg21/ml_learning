"""
Lite smoke tests for local (sliding-window) attention in v2: checks the
properties that make it a window rather than just a different mask - the
default is full attention, tokens outside the window cannot influence a
position while tokens inside it can, it composes with the decoder's causal
mask, and it refuses cross-attention. End-to-end training with a windowed
transformer_v2 is covered in test_smoke_training.py.
"""

from __future__ import annotations

import pytest
import torch

from src.models.v1.modules_v1 import RotaryEmbedding
from src.models.v2.modules_v2 import GroupedQueryAttentionV2


D_MODEL = 32
NUM_HEADS = 4
HEAD_DIM = D_MODEL // NUM_HEADS
SEQ_LEN = 8


def _inputs() -> torch.Tensor:

    torch.manual_seed(0)

    return torch.randn(2, SEQ_LEN, D_MODEL)


def _attention(window_size: int | None) -> GroupedQueryAttentionV2:

    torch.manual_seed(1)

    return GroupedQueryAttentionV2(
        D_MODEL,
        NUM_HEADS,
        num_kv_heads=2,
        dropout_param=0.0,
        use_rope=True,
        window_size=window_size,
    )


def test_window_larger_than_sequence_matches_full_attention():

    rope = RotaryEmbedding(head_dim=HEAD_DIM, max_seq_len=16)

    x = _inputs()

    full = _attention(window_size=None)
    windowed = _attention(window_size=SEQ_LEN)

    assert torch.allclose(
        full(x, x, rope=rope),
        windowed(x, x, rope=rope),
        atol=1e-6,
    )


def test_tokens_outside_the_window_do_not_affect_a_position():

    rope = RotaryEmbedding(head_dim=HEAD_DIM, max_seq_len=16)

    attention = _attention(window_size=3)

    x = _inputs()
    perturbed = x.clone()
    perturbed[:, 7] += 1.0

    baseline = attention(x, x, rope=rope)
    changed = attention(perturbed, perturbed, rope=rope)

    # Position 4 is 3 away from position 7 (outside a window of 3);
    # position 5 is 2 away (inside it).
    assert torch.allclose(baseline[:, :5], changed[:, :5], atol=1e-6)
    assert not torch.allclose(baseline[:, 5], changed[:, 5], atol=1e-6)


def test_window_composes_with_causal_mask():

    rope = RotaryEmbedding(head_dim=HEAD_DIM, max_seq_len=16)

    attention = _attention(window_size=3)

    causal_mask = torch.triu(
        torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool),
        diagonal=1,
    )

    x = _inputs()
    perturbed = x.clone()
    perturbed[:, 5] += 1.0

    baseline = attention(x, x, rope=rope, attn_mask=causal_mask)
    changed = attention(perturbed, perturbed, rope=rope, attn_mask=causal_mask)

    # Causal: nothing before 5 moves. Window of 3: position 7 still
    # reaches back to token 5 (distance 2).
    assert torch.allclose(baseline[:, :5], changed[:, :5], atol=1e-6)
    assert not torch.allclose(baseline[:, 7], changed[:, 7], atol=1e-6)


def test_window_with_padding_mask_stays_finite():

    rope = RotaryEmbedding(head_dim=HEAD_DIM, max_seq_len=16)

    attention = _attention(window_size=2)

    padding_mask = torch.zeros(2, SEQ_LEN, dtype=torch.bool)
    padding_mask[:, 5:] = True

    x = _inputs()

    output = attention(x, x, rope=rope, key_padding_mask=padding_mask)

    assert torch.isfinite(output).all()


def test_window_rejects_cross_attention():

    attention = GroupedQueryAttentionV2(
        D_MODEL,
        NUM_HEADS,
        window_size=3,
    )

    with pytest.raises(ValueError):
        attention(_inputs(), torch.randn(2, 5, D_MODEL))


def test_window_must_be_positive():

    with pytest.raises(ValueError):
        GroupedQueryAttentionV2(D_MODEL, NUM_HEADS, window_size=0)
