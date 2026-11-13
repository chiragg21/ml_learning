"""
Lite smoke tests for post-training weight quantization: int4 packing
round-trips, dequantized weights stay within half a step of the originals,
the model shrinks by the expected factor, outputs barely move at int8
(and move more at int4), and a quantized model can be saved and restored.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.v3.datamodels_v3 import (
    DecoderConfigV3,
    EncoderConfigV3,
    TransformerConfigV3,
)
from src.models.v3.transformer_v3 import TransformerV3
from src.quantization.quantize import (
    QuantizedLinear,
    count_quantized_layers,
    model_size_bytes,
    quantize_model,
)


def _model() -> TransformerV3:

    torch.manual_seed(0)

    parameters = dict(
        max_seq_len=24, d_model=64, num_heads=4, d_ff=128,
        num_layers=2, num_kv_heads=2, dropout_param=0.0,
        num_experts=2, top_k=1, dynamic_depth=True,
    )

    return TransformerV3(
        TransformerConfigV3(
            encoder_config=EncoderConfigV3(vocab_size=40, **parameters),
            decoder_config=DecoderConfigV3(target_vocab_size=40, **parameters),
        )
    ).eval()


def _inputs():

    torch.manual_seed(1)

    return torch.randint(1, 40, (2, 10)), torch.randint(1, 40, (2, 8))


def test_int4_packing_round_trips_every_value():

    values = torch.arange(-7, 8, dtype=torch.int8).repeat(2, 2)[:, :30]

    packed = QuantizedLinear._pack_int4(values)

    assert packed.dtype == torch.uint8
    assert packed.shape == (2, 15)
    assert torch.equal(QuantizedLinear._unpack_int4(packed), values)


@pytest.mark.parametrize("bits,group_size", [(8, None), (8, 16), (4, None), (4, 16)])
def test_dequantized_weights_are_within_half_a_step(bits, group_size):

    torch.manual_seed(0)

    linear = nn.Linear(64, 32, bias=False)
    quantized = QuantizedLinear.from_linear(linear, bits=bits, group_size=group_size)

    error = (quantized.dequantize() - linear.weight).abs()

    step = quantized.scales.repeat_interleave(quantized.group_size, dim=1)

    assert torch.all(error <= step / 2 + 1e-6)


def test_smaller_groups_are_more_accurate():

    torch.manual_seed(0)

    linear = nn.Linear(128, 32, bias=False)

    def error(group_size):
        q = QuantizedLinear.from_linear(linear, bits=4, group_size=group_size)
        return (q.dequantize() - linear.weight).abs().mean()

    assert error(16) < error(None)


def test_quantized_linear_matches_float_linear_closely():

    torch.manual_seed(0)

    linear = nn.Linear(64, 32, bias=False)
    quantized = QuantizedLinear.from_linear(linear, bits=8)

    x = torch.randn(5, 64)

    assert torch.allclose(quantized(x), linear(x), atol=0.05)


def test_invalid_settings_are_rejected():

    with pytest.raises(ValueError):
        QuantizedLinear.from_linear(nn.Linear(8, 8, bias=False), bits=3)

    with pytest.raises(ValueError):
        QuantizedLinear.from_linear(nn.Linear(8, 8), bits=8)

    with pytest.raises(ValueError):
        QuantizedLinear.from_linear(nn.Linear(10, 8, bias=False), bits=8, group_size=4)


def test_quantize_model_copies_and_skips_sensitive_layers():

    model = _model()
    before = model.llm_head.weight.clone()

    quantized = quantize_model(model, bits=8)

    assert count_quantized_layers(model) == 0
    assert count_quantized_layers(quantized) > 0

    assert isinstance(quantized.llm_head, nn.Linear)

    for name, module in quantized.named_modules():
        if "router" in name and isinstance(module, (nn.Linear, QuantizedLinear)):
            assert isinstance(module, nn.Linear)

    assert torch.equal(model.llm_head.weight, before)
    assert not any(p.requires_grad for p in quantized.parameters())


@pytest.mark.parametrize("bits,min_ratio", [(8, 1.5), (4, 2.0)])
def test_quantization_shrinks_the_model(bits, min_ratio):

    model = _model()

    quantized = quantize_model(model, bits=bits, group_size=32)

    assert model_size_bytes(model) / model_size_bytes(quantized) > min_ratio


def test_int8_changes_outputs_less_than_int4():

    model = _model()
    source, target = _inputs()

    with torch.no_grad():
        reference = model(source_ids=source, target_ids=target).logits

        error = {}

        for bits in (8, 4):
            logits = quantize_model(model, bits=bits, group_size=32)(
                source_ids=source, target_ids=target
            ).logits
            error[bits] = (logits - reference).abs().mean().item()

    assert error[8] < error[4]
    assert error[8] < 0.05 * reference.abs().mean().item() + 1e-3


def test_quantized_model_runs_cached_generation_steps():

    model = quantize_model(_model(), bits=8)
    source, target = _inputs()

    with torch.no_grad():
        full = model(source_ids=source, target_ids=target).logits

        memory = model.encode(source)
        cache = model.init_cache()

        steps = torch.cat(
            [model.decode_step(target[:, i : i + 1], memory, cache) for i in range(8)],
            dim=1,
        )

    assert torch.allclose(full, steps, atol=1e-4)


def test_quantized_state_dict_restores_into_a_fresh_quantized_model():

    model = _model()
    quantized = quantize_model(model, bits=4, group_size=32)

    restored = quantize_model(_model(), bits=4, group_size=32)

    # perturb so the test fails if loading does nothing
    for name, buffer in restored.named_buffers():
        if name.endswith("scales"):
            buffer.add_(1.0)

    restored.load_state_dict(quantized.state_dict())

    source, target = _inputs()

    with torch.no_grad():
        assert torch.equal(
            quantized(source_ids=source, target_ids=target).logits,
            restored(source_ids=source, target_ids=target).logits,
        )
