from __future__ import annotations

from dataclasses import dataclass

import torch

from ..v2.datamodels_v2 import (
    DecoderConfigV2,
    EncoderConfigV2,
    LayerConfigV2,
)


# The fields below are v3's additions; everything else (GQA, window,
# banded, ...) is inherited from v2 unchanged.

@dataclass
class LayerConfigV3(LayerConfigV2):
    # None = ordinary dense SwiGLU feed-forward in this layer.
    num_experts: int | None = None
    top_k: int = 2
    moe_aux_weight: float = 0.01
    dynamic_depth: bool = False
    depth_capacity: float = 0.5
    depth_aux_weight: float = 0.01


@dataclass
class EncoderConfigV3(EncoderConfigV2):
    # MoE replaces the dense FFN in every `moe_every`-th layer (1 = all,
    # 2 = every other layer, counting from the last of each group).
    num_experts: int | None = None
    top_k: int = 2
    moe_every: int = 1
    moe_aux_weight: float = 0.01
    dynamic_depth: bool = False
    depth_capacity: float = 0.5
    depth_aux_weight: float = 0.01


@dataclass
class DecoderConfigV3(DecoderConfigV2):
    num_experts: int | None = None
    top_k: int = 2
    moe_every: int = 1
    moe_aux_weight: float = 0.01
    dynamic_depth: bool = False
    depth_capacity: float = 0.5
    depth_aux_weight: float = 0.01


@dataclass
class TransformerConfigV3:
    encoder_config: EncoderConfigV3
    decoder_config: DecoderConfigV3


@dataclass
class TransformerOutput:
    logits: torch.Tensor
    # Already weighted (MoE load balancing + depth budget); the trainer
    # just adds it to the cross-entropy loss. Zero for a dense model.
    aux_loss: torch.Tensor | None = None
