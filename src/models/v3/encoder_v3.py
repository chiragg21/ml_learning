from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..v1.modules_v1 import RMSNorm, RotaryEmbedding, SwiGLUFeedForward
from ..v2.encoder_v2 import EncoderV2
from ..v2.modules_v2 import GroupedQueryAttentionV2
from .datamodels_v3 import EncoderConfigV3, LayerConfigV3
from .modules_v3 import DepthRouter, MoEFeedForward

"""
Encoder layer (v3): v2's layer with two optional changes.

  X1 = X + SelfAttn(RMSNorm(X))                       (GQA + RoPE, as v2)
  X2 = X1 + FFN(RMSNorm(X1))       FFN = SwiGLU, or MoEFeedForward
  out = X + gate * (X2 - X)        gate from DepthRouter, if dynamic_depth

With num_experts=None and dynamic_depth=False this is exactly v2's layer
(same parameters, same maths). The layer returns (output, aux_loss) so
the MoE load-balancing and depth-budget losses can reach the trainer.

Extra learnable parameters per layer over v2:
 - MoE layer: D*E router + (E-1)*3*D*F (E experts instead of one FFN)
 - dynamic depth: 2*D + 1
"""


class V3LayerMixin:
    """Feed-forward and depth-gating logic shared by encoder/decoder layers."""

    def _build_extras(self, config: LayerConfigV3) -> None:

        self.moe_aux_weight = config.moe_aux_weight
        self.depth_aux_weight = config.depth_aux_weight

        if config.num_experts is None:
            self.ffn = SwiGLUFeedForward(
                d_model=config.d_model,
                d_ff=config.d_ff,
                dropout_param=config.dropout_param,
            )
            self.is_moe = False
        else:
            self.ffn = MoEFeedForward(
                d_model=config.d_model,
                d_ff=config.d_ff,
                num_experts=config.num_experts,
                top_k=config.top_k,
                dropout_param=config.dropout_param,
            )
            self.is_moe = True

        self.depth_router = (
            DepthRouter(config.d_model, config.depth_capacity)
            if config.dynamic_depth
            else None
        )

    def _feed_forward(
        self,
        normed: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if not self.is_moe:
            return self.ffn(normed), normed.new_zeros(())

        output, balance_loss = self.ffn(normed, token_mask)

        return output, self.moe_aux_weight * balance_loss

    def _apply_depth(
        self,
        layer_input: torch.Tensor,
        layer_output: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.depth_router is None:
            return layer_output, layer_output.new_zeros(())

        gate, budget_loss = self.depth_router(layer_input, token_mask)

        gate = gate.to(layer_output.dtype)

        return (
            layer_input + gate * (layer_output - layer_input),
            self.depth_aux_weight * budget_loss,
        )


class EncoderLayerV3(V3LayerMixin, nn.Module):

    def __init__(
            self,
            config: LayerConfigV3,
            rope: RotaryEmbedding,
    ):
        super().__init__()

        self.self_attention = GroupedQueryAttentionV2(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout_param=config.dropout_param,
            use_rope=True,
            window_size=config.window_size,
            banded=config.banded,
        )

        self._build_extras(config)

        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout_param)

        self.rope = rope

    def forward(self, x, key_padding_mask=None, segment_mask=None):

        token_mask = None if key_padding_mask is None else ~key_padding_mask

        layer_input = x

        normed = self.norm1(x)

        attention_output = self.self_attention(
            normed,
            normed,
            rope=self.rope,
            key_padding_mask=key_padding_mask,
            segment_mask=segment_mask,
        )

        x = x + self.dropout(attention_output)

        normed = self.norm2(x)

        ffn_output, aux_loss = self._feed_forward(normed, token_mask)

        x = x + self.dropout(ffn_output)

        x, depth_loss = self._apply_depth(layer_input, x, token_mask)

        return x, aux_loss + depth_loss


class EncoderV3(EncoderV2):
    """
    EncoderV2 with v3 layers. Embedding, RoPE and final norm come from v2
    unchanged; the v2 layer stack its constructor builds is replaced.
    forward() returns (memory, aux_loss).
    """

    def __init__(
            self,
            config: EncoderConfigV3,
        ):

        super().__init__(config)

        self.layers = nn.ModuleList([
            EncoderLayerV3(
                LayerConfigV3(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    window_size=self.window_size,
                    banded=self.banded,
                    d_ff=self.d_ff,
                    dropout_param=self.dropout_param,
                    rope_theta=config.rope_theta,
                    num_experts=(
                        config.num_experts
                        if (index + 1) % config.moe_every == 0
                        else None
                    ),
                    top_k=config.top_k,
                    moe_aux_weight=config.moe_aux_weight,
                    dynamic_depth=config.dynamic_depth,
                    depth_capacity=config.depth_capacity,
                    depth_aux_weight=config.depth_aux_weight,
                ),
                rope=self.rope,
            )
            for index in range(self.num_layers)
        ])

    def forward(
            self,
            tokens,
            key_padding_mask=None,
            segment_mask=None,
        ):

        seq_len = tokens.shape[1]

        if seq_len > self.max_seq_len:
            raise ValueError(
                "Sequence Length exceeds max_seq_len"
            )

        x = self.dropout(self.token_embedding(tokens))

        aux_loss = x.new_zeros(())

        for layer in self.layers:

            if self.gradient_checkpointing and self.training:
                x, layer_aux = checkpoint(
                    layer,
                    x,
                    key_padding_mask,
                    segment_mask,
                    use_reentrant=False,
                )
            else:
                x, layer_aux = layer(
                    x,
                    key_padding_mask=key_padding_mask,
                    segment_mask=segment_mask,
                )

            aux_loss = aux_loss + layer_aux

        return self.final_norm(x), aux_loss
