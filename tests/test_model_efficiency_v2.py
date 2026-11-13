"""
Lite smoke tests for v2's efficiency features at the model level: the
banded kernel matches the masked path, the KV cache reproduces the full
forward pass, packed rows behave like separate examples, and gradient
checkpointing leaves gradients unchanged. Training end-to-end with these
turned on is covered in test_smoke_training.py.
"""

from __future__ import annotations

import pytest
import torch

from src.models.v1.modules_v1 import RotaryEmbedding
from src.models.v2.datamodels_v2 import (
    DecoderConfigV2,
    EncoderConfigV2,
    TransformerConfigV2,
)
from src.models.v2.modules_v2 import GroupedQueryAttentionV2
from src.models.v2.transformer_v2 import TransformerV2


VOCAB = 30
PAD = 0


def _model(window_size=None, banded=False) -> TransformerV2:

    torch.manual_seed(0)

    parameters = dict(
        max_seq_len=24,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        num_kv_heads=2,
        window_size=window_size,
        banded=banded,
        dropout_param=0.0,
    )

    model = TransformerV2(
        TransformerConfigV2(
            encoder_config=EncoderConfigV2(vocab_size=VOCAB, **parameters),
            decoder_config=DecoderConfigV2(target_vocab_size=VOCAB, **parameters),
        )
    )

    return model.eval()


def _batch(batch_size=2, source_len=10, target_len=8):

    torch.manual_seed(1)

    source = torch.randint(1, VOCAB, (batch_size, source_len))
    target = torch.randint(1, VOCAB, (batch_size, target_len))

    # right padding on the second row, to exercise the padding masks
    source[1, -3:] = PAD
    target[1, -2:] = PAD

    return source, target


# ==================================================
# Banded kernel
# ==================================================

@pytest.mark.parametrize("window_size", [1, 3, 5, 20])
@pytest.mark.parametrize("causal", [False, True])
def test_banded_attention_matches_masked_attention(window_size, causal):

    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)

    def attention(banded):

        torch.manual_seed(2)

        return GroupedQueryAttentionV2(
            32, 4, num_kv_heads=2, dropout_param=0.0, use_rope=True,
            window_size=window_size, banded=banded,
        )

    masked, banded = attention(False), attention(True)

    torch.manual_seed(3)
    x = torch.randn(2, 12, 32)

    attn_mask = torch.triu(torch.ones(12, 12, dtype=torch.bool), 1) if causal else None
    padding = torch.zeros(2, 12, dtype=torch.bool)
    padding[1, -4:] = True

    expected = masked(x, x, rope=rope, attn_mask=attn_mask, key_padding_mask=padding)
    actual = banded(x, x, rope=rope, attn_mask=attn_mask, key_padding_mask=padding)

    valid = ~padding[:, :, None]

    assert torch.allclose(expected * valid, actual * valid, atol=1e-5)


def test_banded_transformer_matches_masked_transformer():

    source, target = _batch()

    reference = _model(window_size=3)
    banded = _model(window_size=3, banded=True)
    banded.load_state_dict(reference.state_dict())

    kwargs = dict(
        source_ids=source,
        target_ids=target,
        source_key_padding_mask=source == PAD,
        target_key_padding_mask=target == PAD,
    )

    expected = reference(**kwargs).logits
    actual = banded(**kwargs).logits

    keep = (target != PAD)[:, :, None]

    assert torch.allclose(expected * keep, actual * keep, atol=1e-5)


def test_banded_requires_a_window():

    with pytest.raises(ValueError):
        GroupedQueryAttentionV2(32, 4, banded=True)


# ==================================================
# KV cache
# ==================================================

@pytest.mark.parametrize("window_size", [None, 3])
def test_kv_cache_reproduces_full_forward(window_size):

    model = _model(window_size=window_size)
    source, target = _batch()

    source_padding = source == PAD

    with torch.no_grad():

        full = model(
            source_ids=source,
            target_ids=target,
            source_key_padding_mask=source_padding,
        ).logits

        memory = model.encode(source, source_padding)
        cache = model.init_cache()

        steps = [
            model.decode_step(
                target[:, i : i + 1],
                memory,
                cache,
                memory_key_padding_mask=source_padding,
            )
            for i in range(target.shape[1])
        ]

    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5)


def test_kv_cache_prefill_of_several_tokens_matches_stepwise():

    model = _model()
    source, target = _batch()

    with torch.no_grad():

        memory = model.encode(source)

        prefill_cache = model.init_cache()
        prefill = model.decode_step(target[:, :5], memory, prefill_cache)
        next_prefill = model.decode_step(target[:, 5:6], memory, prefill_cache)

        full = model(source_ids=source, target_ids=target).logits

    assert torch.allclose(prefill, full[:, :5], atol=1e-5)
    assert torch.allclose(next_prefill, full[:, 5:6], atol=1e-5)


def test_window_cache_stays_bounded():

    model = _model(window_size=3)
    source, target = _batch()

    with torch.no_grad():

        memory = model.encode(source)
        cache = model.init_cache()

        for i in range(target.shape[1]):
            model.decode_step(target[:, i : i + 1], memory, cache)

    # window 3 -> only the 2 previous keys are ever still in reach
    assert cache[0]["self"]["k"].shape[-2] == 3
    assert cache[0]["self"]["seen"] == target.shape[1]


def test_cache_rejects_sequences_past_max_seq_len():

    model = _model()
    source, _ = _batch()

    memory = model.encode(source)
    cache = model.init_cache()

    with pytest.raises(ValueError):
        for _ in range(25):
            model.decode_step(torch.ones(2, 1, dtype=torch.long), memory, cache)


# ==================================================
# Packing
# ==================================================

def test_packed_row_matches_separate_examples():

    model = _model()

    torch.manual_seed(4)

    source_a = torch.randint(1, VOCAB, (1, 5))
    source_b = torch.randint(1, VOCAB, (1, 4))
    target_a = torch.randint(1, VOCAB, (1, 4))
    target_b = torch.randint(1, VOCAB, (1, 6))

    with torch.no_grad():

        alone_a = model(source_ids=source_a, target_ids=target_a).logits
        alone_b = model(source_ids=source_b, target_ids=target_b).logits

        # one row, padded at the end, two segments (0 is reserved for pad)
        source = torch.cat([source_a, source_b, torch.zeros(1, 2, dtype=torch.long)], 1)
        target = torch.cat([target_a, target_b, torch.zeros(1, 1, dtype=torch.long)], 1)

        source_segments = torch.tensor([[1] * 5 + [2] * 4 + [0] * 2])
        target_segments = torch.tensor([[1] * 4 + [2] * 6 + [0] * 1])

        packed = model(
            source_ids=source,
            target_ids=target,
            source_key_padding_mask=source == PAD,
            target_key_padding_mask=target == PAD,
            source_segment_ids=source_segments,
            target_segment_ids=target_segments,
        ).logits

    assert torch.isfinite(packed).all()
    assert torch.allclose(packed[:, :4], alone_a, atol=1e-5)
    assert torch.allclose(packed[:, 4:10], alone_b, atol=1e-5)


def test_packing_works_with_banded_attention():

    reference = _model(window_size=3)
    banded = _model(window_size=3, banded=True)
    banded.load_state_dict(reference.state_dict())

    source = torch.randint(1, VOCAB, (1, 8))
    target = torch.randint(1, VOCAB, (1, 8))

    segments = torch.tensor([[1] * 4 + [2] * 4])

    kwargs = dict(
        source_ids=source,
        target_ids=target,
        source_segment_ids=segments,
        target_segment_ids=segments,
    )

    with torch.no_grad():
        assert torch.allclose(
            reference(**kwargs).logits,
            banded(**kwargs).logits,
            atol=1e-5,
        )


def test_segment_ids_must_come_in_pairs():

    model = _model()
    source, target = _batch()

    with pytest.raises(ValueError):
        model(source_ids=source, target_ids=target, source_segment_ids=torch.ones_like(source))


# ==================================================
# Gradient checkpointing
# ==================================================

def test_gradient_checkpointing_preserves_gradients():

    source, target = _batch()

    def gradients(checkpointing):

        model = _model().train()
        model.set_gradient_checkpointing(checkpointing)

        logits = model(source_ids=source, target_ids=target).logits
        logits.square().mean().backward()

        return [p.grad.clone() for p in model.parameters() if p.grad is not None]

    plain, checkpointed = gradients(False), gradients(True)

    assert len(plain) == len(checkpointed)

    for expected, actual in zip(plain, checkpointed):
        assert torch.allclose(expected, actual, atol=1e-6)
