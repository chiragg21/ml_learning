from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..v1.modules_v1 import RMSNorm, RotaryEmbedding
from ..v2.decoder_v2 import DecoderV2
from ..v2.modules_v2 import GroupedQueryAttentionV2
from .datamodels_v3 import DecoderConfigV3, LayerConfigV3
from .encoder_v3 import V3LayerMixin

"""
Decoder layer (v3): v2's layer with the optional MoE feed-forward and
depth gate described in encoder_v3.py.

  X1 = X  + SelfAttn(RMSNorm(X), causal mask)
  X2 = X1 + CrossAttn(RMSNorm(X1), memory)
  X3 = X2 + FFN(RMSNorm(X2))        FFN = SwiGLU, or MoEFeedForward
  out = X + gate * (X3 - X)         if dynamic_depth

Both extras act per token, so they work unchanged with the KV cache: the
decoder still feeds one new token per step and every token's route
depends only on that token.
"""


class DecoderLayerV3(V3LayerMixin, nn.Module):

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

        self.cross_attention = GroupedQueryAttentionV2(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout_param=config.dropout_param,
            use_rope=False,
            cross_attention=True,
        )

        self._build_extras(config)

        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.norm3 = RMSNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout_param)

        self.rope = rope

    def forward(
            self,
            x,
            memory,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            self_segment_mask=None,
            cross_segment_mask=None,
            cache=None,
    ):

        token_mask = (
            None if tgt_key_padding_mask is None else ~tgt_key_padding_mask
        )

        layer_input = x

        normed = self.norm1(x)

        x = x + self.dropout(self.self_attention(
            normed,
            normed,
            rope=self.rope,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            segment_mask=self_segment_mask,
            cache=None if cache is None else cache["self"],
        ))

        normed = self.norm2(x)

        x = x + self.dropout(self.cross_attention(
            normed,
            memory,
            key_padding_mask=memory_key_padding_mask,
            segment_mask=cross_segment_mask,
            cache=None if cache is None else cache["cross"],
        ))

        normed = self.norm3(x)

        ffn_output, aux_loss = self._feed_forward(normed, token_mask)

        x = x + self.dropout(ffn_output)

        x, depth_loss = self._apply_depth(layer_input, x, token_mask)

        return x, aux_loss + depth_loss


class DecoderV3(DecoderV2):
    """
    DecoderV2 with v3 layers; forward() returns (hidden, aux_loss).
    """

    def __init__(
            self,
            config: DecoderConfigV3,
        ):

        super().__init__(config)

        self.layers = nn.ModuleList([
            DecoderLayerV3(
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
            target_ids,
            memory,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            self_segment_mask=None,
            cross_segment_mask=None,
            cache=None,
    ):
        target_seq_len = target_ids.shape[1]

        past_len = 0 if cache is None else cache[0]["self"].get("seen", 0)

        if past_len + target_seq_len > self.max_seq_len:
            raise ValueError(
                "Target seq len is greater than max seq len"
            )

        x = self.dropout(self.token_embeddings(target_ids))

        if tgt_mask is None and cache is None:
            tgt_mask = torch.triu(
                torch.ones(
                    target_seq_len,
                    target_seq_len,
                    device=target_ids.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )

        aux_loss = x.new_zeros(())

        for layer_index, layer in enumerate(self.layers):

            if self.gradient_checkpointing and self.training:
                x, layer_aux = checkpoint(
                    layer,
                    x,
                    memory,
                    tgt_mask,
                    tgt_key_padding_mask,
                    memory_key_padding_mask,
                    self_segment_mask,
                    cross_segment_mask,
                    use_reentrant=False,
                )
            else:
                x, layer_aux = layer(
                    x,
                    memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    self_segment_mask=self_segment_mask,
                    cross_segment_mask=cross_segment_mask,
                    cache=None if cache is None else cache[layer_index],
                )

            aux_loss = aux_loss + layer_aux

        return self.final_norm(x), aux_loss
