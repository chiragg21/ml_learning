"""
Lite smoke tests for transformer_v3: with the new features off it is
exactly v2, the MoE layer routes/balances/trains as described, dynamic
depth gates tokens, and the v2 machinery (KV cache, packing, gradient
checkpointing) still works with both on. End-to-end training is covered
in test_smoke_training.py.
"""

from __future__ import annotations

import pytest
import torch

from src.models.v2.datamodels_v2 import (
    DecoderConfigV2,
    EncoderConfigV2,
    TransformerConfigV2,
)
from src.models.v2.transformer_v2 import TransformerV2
from src.models.v3.datamodels_v3 import (
    DecoderConfigV3,
    EncoderConfigV3,
    TransformerConfigV3,
)
from src.models.v3.modules_v3 import DepthRouter, MoEFeedForward
from src.models.v3.transformer_v3 import TransformerV3


VOCAB = 30
PAD = 0

BASE = dict(
    max_seq_len=24,
    d_model=32,
    num_heads=4,
    d_ff=64,
    num_layers=2,
    num_kv_heads=2,
    dropout_param=0.0,
)


def _v3(**extra) -> TransformerV3:

    torch.manual_seed(0)

    return TransformerV3(
        TransformerConfigV3(
            encoder_config=EncoderConfigV3(vocab_size=VOCAB, **BASE, **extra),
            decoder_config=DecoderConfigV3(target_vocab_size=VOCAB, **BASE, **extra),
        )
    )


def _batch():

    torch.manual_seed(1)

    source = torch.randint(1, VOCAB, (2, 10))
    target = torch.randint(1, VOCAB, (2, 8))

    source[1, -3:] = PAD
    target[1, -2:] = PAD

    return source, target


def _kwargs(source, target):

    return dict(
        source_ids=source,
        target_ids=target,
        source_key_padding_mask=source == PAD,
        target_key_padding_mask=target == PAD,
    )


# ==================================================
# Dense limit
# ==================================================

def test_v3_without_new_features_equals_v2():

    v3 = _v3().eval()

    torch.manual_seed(0)

    v2 = TransformerV2(
        TransformerConfigV2(
            encoder_config=EncoderConfigV2(vocab_size=VOCAB, **BASE),
            decoder_config=DecoderConfigV2(target_vocab_size=VOCAB, **BASE),
        )
    ).eval()

    v2.load_state_dict(v3.state_dict())

    source, target = _batch()

    assert torch.allclose(
        v3(**_kwargs(source, target)).logits,
        v2(**_kwargs(source, target)).logits,
        atol=1e-6,
    )
    assert v3(**_kwargs(source, target)).aux_loss == 0


# ==================================================
# Mixture of experts
# ==================================================

def test_moe_output_is_the_weighted_sum_of_the_chosen_experts():

    torch.manual_seed(0)

    moe = MoEFeedForward(16, 32, num_experts=4, top_k=2, dropout_param=0.0).eval()

    x = torch.randn(2, 5, 16)

    output, _ = moe(x)

    flat = x.reshape(-1, 16)
    probs = torch.softmax(moe.router(flat), dim=-1)
    top_probs, top_experts = probs.topk(2, dim=-1)
    top_probs = top_probs / top_probs.sum(-1, keepdim=True)

    expected = torch.zeros_like(flat)

    for token in range(flat.shape[0]):
        for slot in range(2):
            expert = moe.experts[top_experts[token, slot]]
            expected[token] += top_probs[token, slot] * expert(flat[token])

    assert torch.allclose(output.reshape(-1, 16), expected, atol=1e-5)


def test_moe_top_k_equal_to_num_experts_uses_every_expert():

    moe = MoEFeedForward(16, 32, num_experts=3, top_k=3).eval()
    moe.track_usage = True

    moe(torch.randn(2, 6, 16))

    assert torch.allclose(moe.last_usage, torch.full((3,), 1 / 3))


def test_balance_loss_is_one_when_balanced_and_larger_when_collapsed():

    moe = MoEFeedForward(16, 32, num_experts=4, top_k=1)

    x = torch.randn(1, 64, 16)

    # collapsed: the router sends everything to expert 0
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0] = 5.0

    _, collapsed = moe(x.abs())

    # uniform router: every probability is exactly 1/E
    with torch.no_grad():
        moe.router.weight.zero_()

    _, uniform = moe(x)

    assert collapsed > 2.0
    assert torch.isclose(uniform, torch.tensor(1.0), atol=0.3)


def test_balance_loss_ignores_padding_tokens():

    torch.manual_seed(0)

    moe = MoEFeedForward(16, 32, num_experts=4, top_k=2).eval()

    x = torch.randn(1, 10, 16)
    mask = torch.tensor([[True] * 6 + [False] * 4])

    _, masked = moe(x, mask)
    _, trimmed = moe(x[:, :6])

    assert torch.isclose(masked, trimmed, atol=1e-5)


def test_moe_validates_top_k():

    with pytest.raises(ValueError):
        MoEFeedForward(16, 32, num_experts=2, top_k=3)


def test_moe_transformer_reports_an_aux_loss_and_every_expert_gets_gradient():

    model = _v3(num_experts=4, top_k=2).train()

    source, target = _batch()

    output = model(**_kwargs(source, target))

    assert output.aux_loss > 0

    (output.logits.square().mean() + output.aux_loss).backward()

    router_grads = [
        p.grad for n, p in model.named_parameters() if "router" in n
    ]

    assert router_grads and all(g is not None and g.abs().sum() > 0 for g in router_grads)


def test_moe_every_only_converts_some_layers():

    model = _v3(num_experts=4, moe_every=2)

    assert [layer.is_moe for layer in model.encoder.layers] == [False, True]


def test_moe_routing_usage_is_reported():

    model = _v3(num_experts=4, top_k=2).eval()
    model.set_usage_tracking(True)

    source, target = _batch()
    model(**_kwargs(source, target))

    usage = model.routing_usage()

    assert usage
    assert all(abs(v.sum().item() - 1.0) < 1e-5 for v in usage.values())


# ==================================================
# Dynamic depth
# ==================================================

def test_depth_router_starts_open_and_gates_per_token():

    router = DepthRouter(16, capacity=0.5)
    router.initialize()

    gate, loss = router(torch.randn(2, 5, 16))

    assert gate.shape == (2, 5, 1)
    assert torch.all(gate == 1)
    assert loss > 0  # mean p ~ 0.88 against a capacity of 0.5

    with torch.no_grad():
        router.score.bias.fill_(-5.0)

    gate, _ = router(torch.randn(2, 5, 16))

    assert torch.all(gate == 0)


def test_depth_router_gradient_flows_through_the_hard_gate():

    router = DepthRouter(16)
    router.initialize()

    gate, _ = router(torch.randn(2, 5, 16))
    gate.sum().backward()

    assert router.score.bias.grad is not None and router.score.bias.grad != 0


def test_skipped_layer_leaves_tokens_unchanged():

    model = _v3(dynamic_depth=True).eval()

    for module in model.modules():
        if isinstance(module, DepthRouter):
            with torch.no_grad():
                module.score.bias.fill_(-5.0)

    source, target = _batch()

    # every layer skipped -> only the embedding and final norm remain
    encoded = model.encode(source)

    expected = model.encoder.final_norm(model.encoder.token_embedding(source))

    assert torch.allclose(encoded, expected, atol=1e-6)


def test_depth_budget_loss_pushes_usage_toward_capacity():

    router = DepthRouter(8, capacity=0.25)
    router.initialize()

    optimizer = torch.optim.SGD(router.parameters(), lr=1.0)

    x = torch.randn(4, 16, 8)

    for _ in range(200):
        _, loss = router(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    soft = torch.sigmoid(router.score(router.norm(x))).mean()

    assert abs(soft.item() - 0.25) < 0.05


def test_depth_capacity_is_validated():

    with pytest.raises(ValueError):
        DepthRouter(8, capacity=0.0)


# ==================================================
# v2 machinery still works with v3 layers
# ==================================================

KINDS = dict(num_experts=4, top_k=2, dynamic_depth=True)


def test_kv_cache_matches_full_forward_with_moe_and_depth():

    model = _v3(**KINDS).eval()

    source, target = _batch()
    padding = source == PAD

    with torch.no_grad():

        full = model(source_ids=source, target_ids=target,
                     source_key_padding_mask=padding).logits

        memory = model.encode(source, padding)
        cache = model.init_cache()

        steps = [
            model.decode_step(target[:, i : i + 1], memory, cache,
                              memory_key_padding_mask=padding)
            for i in range(target.shape[1])
        ]

    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5)


def test_gradient_checkpointing_preserves_v3_gradients():

    source, target = _batch()

    def gradients(checkpointing):

        model = _v3(**KINDS).train()
        model.set_gradient_checkpointing(checkpointing)

        output = model(**_kwargs(source, target))
        (output.logits.square().mean() + output.aux_loss).backward()

        return [p.grad.clone() for p in model.parameters() if p.grad is not None]

    for expected, actual in zip(gradients(False), gradients(True)):
        assert torch.allclose(expected, actual, atol=1e-5)


def test_packed_row_matches_separate_examples_with_moe():

    model = _v3(num_experts=4, top_k=2).eval()

    torch.manual_seed(4)

    sa, sb = torch.randint(1, VOCAB, (1, 5)), torch.randint(1, VOCAB, (1, 4))
    ta, tb = torch.randint(1, VOCAB, (1, 4)), torch.randint(1, VOCAB, (1, 6))

    with torch.no_grad():

        alone_a = model(source_ids=sa, target_ids=ta).logits
        alone_b = model(source_ids=sb, target_ids=tb).logits

        packed = model(
            source_ids=torch.cat([sa, sb], 1),
            target_ids=torch.cat([ta, tb], 1),
            source_segment_ids=torch.tensor([[1] * 5 + [2] * 4]),
            target_segment_ids=torch.tensor([[1] * 4 + [2] * 6]),
        ).logits

    assert torch.allclose(packed[:, :4], alone_a, atol=1e-5)
    assert torch.allclose(packed[:, 4:], alone_b, atol=1e-5)
