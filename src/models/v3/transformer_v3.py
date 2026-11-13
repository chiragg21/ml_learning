import torch
from torch import nn

from ..template import BaseTransformer
from ..v2.transformer_v2 import TransformerV2
from .datamodels_v3 import TransformerConfigV3, TransformerOutput
from .decoder_v3 import DecoderV3
from .encoder_v3 import EncoderV3
from .modules_v3 import DepthRouter, MoEFeedForward


class TransformerV3(TransformerV2):
    """
    v3 summary: everything v2 has (GQA, local/banded attention, KV cache,
    packing, gradient checkpointing, ...), plus
      - Mixture-of-Experts feed-forward layers with top-k routing and a
        load-balancing loss (num_experts / top_k / moe_every in the
        encoder/decoder parameters; modules_v3.MoEFeedForward)
      - per-token dynamic depth (dynamic_depth / depth_capacity;
        modules_v3.DepthRouter)

    With neither enabled it computes exactly what TransformerV2 does.
    forward() returns TransformerOutput(logits, aux_loss); aux_loss is the
    already-weighted sum of every layer's balancing and budget losses, and
    Trainer adds it to the cross-entropy. Inference-time compute lives in
    src/generator/inference_compute.py, quantization in
    src/quantization/, and richer evaluation in src/evaluation/advanced.py
    - none of those change the model itself.
    """

    def __init__(
            self,
            config: TransformerConfigV3
    ):
        # Skip TransformerV2.__init__ (it would build v2's encoder and
        # decoder); everything else it sets up is repeated here.
        BaseTransformer.__init__(self, config)

        if config.encoder_config.d_model != config.decoder_config.d_model:
            raise ValueError(
                "Encoder and decoder d_model values must match for cross-attention."
            )

        self.encoder = EncoderV3(config.encoder_config)
        self.decoder = DecoderV3(config.decoder_config)

        self.llm_head = nn.Linear(
            self.decoder.d_model,
            self.decoder.target_vocab_size,
            bias=False,
        )

        # Weight tying, as in v1/v2.
        self.llm_head.weight = self.decoder.token_embeddings.weight

        self.apply(self._init_weights)

        # after _init_weights, which would zero the routers' bias
        for module in self.modules():
            if isinstance(module, DepthRouter):
                module.initialize()

    def forward(
            self,
            source_ids: torch.Tensor,
            target_ids: torch.Tensor,
            source_key_padding_mask: torch.Tensor | None = None,
            target_key_padding_mask: torch.Tensor | None = None,
            source_segment_ids: torch.Tensor | None = None,
            target_segment_ids: torch.Tensor | None = None,
        ):

            source_segment_mask, target_segment_mask, cross_segment_mask = (
                self._segment_masks(source_segment_ids, target_segment_ids)
            )

            encoder_output, encoder_aux = self.encoder(
                 source_ids,
                 key_padding_mask=source_key_padding_mask,
                 segment_mask=source_segment_mask,
            )

            decoder_output, decoder_aux = self.decoder(
                 target_ids=target_ids,
                 memory=encoder_output,
                 tgt_key_padding_mask=target_key_padding_mask,
                 memory_key_padding_mask=source_key_padding_mask,
                 self_segment_mask=target_segment_mask,
                 cross_segment_mask=cross_segment_mask,
            )

            return TransformerOutput(
                 logits=self.llm_head(decoder_output),
                 aux_loss=encoder_aux + decoder_aux,
            )

    # ==================================================
    # Incremental decoding (KV cache)
    # ==================================================

    def encode(
            self,
            source_ids: torch.Tensor,
            source_key_padding_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:

            memory, _ = self.encoder(
                source_ids,
                key_padding_mask=source_key_padding_mask,
            )

            return memory

    def decode_step(
            self,
            new_target_ids: torch.Tensor,
            memory: torch.Tensor,
            cache: list[dict],
            memory_key_padding_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:

            decoder_output, _ = self.decoder(
                target_ids=new_target_ids,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                cache=cache,
            )

            return self.llm_head(decoder_output)

    # ==================================================
    # Diagnostics
    # ==================================================

    def set_usage_tracking(self, enabled: bool) -> None:
        """Make MoE/depth layers record how tokens were routed."""

        for module in self.modules():
            if isinstance(module, (MoEFeedForward, DepthRouter)):
                module.track_usage = enabled

    def routing_usage(self) -> dict[str, torch.Tensor]:
        """
        Last recorded routing per layer (needs set_usage_tracking(True)
        and a forward pass): "<stack>.<layer>.experts" -> fraction of
        routing slots per expert; "<stack>.<layer>.depth" -> fraction of
        tokens that used the layer.
        """

        usage = {}

        for name, module in self.named_modules():

            if isinstance(module, MoEFeedForward) and module.last_usage is not None:
                usage[f"{name}.experts"] = module.last_usage

            if isinstance(module, DepthRouter) and module.last_use_fraction is not None:
                usage[f"{name}.depth"] = module.last_use_fraction

        return usage
