import torch
from torch import nn
from ..template import BaseTransformer
from .datamodels_v2 import TransformerOutput, TransformerConfigV2
from .encoder_v2 import EncoderV2
from .decoder_v2 import DecoderV2


class TransformerV2(BaseTransformer):
    """
    v2 summary: everything v1 has (Pre-Norm, RMSNorm, RoPE, SwiGLU, weight
    tying - the norm, RoPE and FFN modules are reused straight from
    modules_v1.py), plus
      - Grouped-Query Attention in every self- and cross-attention
        layer (modules_v2.py), controlled by num_kv_heads in the
        encoder/decoder config. num_kv_heads=None (or == num_heads)
        reproduces v1's attention exactly.
      - optional local attention (window_size), computed over just the
        band with banded=True
      - a KV cache for generation (init_cache/encode/decode_step)
      - packed-sequence training (source/target_segment_ids)
      - gradient checkpointing (set_gradient_checkpointing)

    The forward() signature is intentionally identical to TransformerV0's
    - same kwargs, same TransformerOutput return type - so nothing in
    Trainer/Evaluator/Predictor/Generator needs to know which version
    it's holding. Switching between v0, v1 and v2 (or between datasets) is
    purely a config change (model.transformer.version, data.name/version
    in an ExperimentConfig) - see configs/ for examples.
    """

    def __init__(
            self,
            config: TransformerConfigV2
    ):
        super().__init__(config)

        if config.encoder_config.d_model != config.decoder_config.d_model:
            raise ValueError(
                "Encoder and decoder d_model values must match for cross-attention."
            )

        self.encoder = EncoderV2(config.encoder_config)
        self.decoder = DecoderV2(config.decoder_config)

        self.llm_head = nn.Linear(
            self.decoder.d_model,
            self.decoder.target_vocab_size,
            bias=False,
        )

        # Weight tying: reuse the decoder's token-embedding matrix as the
        # output projection instead of learning a second, separate V*D
        # matrix. Valid because both live in the same d_model space and
        # share the same vocabulary (the target side's). Saves V*D
        # parameters outright - for this repo's small char vocab (a few
        # hundred tokens) that's tiny, but for a real subword vocab
        # (30k+) it's often 10-20% of a small model's total parameters.
        self.llm_head.weight = self.decoder.token_embeddings.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """
        Same explicit std=0.02 init as TransformerV1 and for the same
        reason: the decoder's embedding matrix doubles as the output
        projection, and nn.Embedding's default Normal(0, 1) is far too
        large for that second role. See transformer_v1.py for the full
        explanation of the loss blow-up this prevents.
        """

        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    # ==================================================
    # Memory-saving switches
    # ==================================================

    def set_gradient_checkpointing(self, enabled: bool) -> None:

        self.encoder.gradient_checkpointing = enabled
        self.decoder.gradient_checkpointing = enabled

    # ==================================================
    # Training / teacher-forced forward
    # ==================================================

    def forward(
            self,
            source_ids: torch.Tensor,
            target_ids: torch.Tensor,
            source_key_padding_mask: torch.Tensor | None = None,
            target_key_padding_mask: torch.Tensor | None = None,
            source_segment_ids: torch.Tensor | None = None,
            target_segment_ids: torch.Tensor | None = None,
        ):

            # Packed rows: an example may only attend within its own
            # segment, in the encoder, the decoder and across them. True
            # = blocked, the same convention as every other mask here.
            source_segment_mask = None
            target_segment_mask = None
            cross_segment_mask = None

            if source_segment_ids is not None:

                if target_segment_ids is None:
                    raise ValueError(
                        "source_segment_ids and target_segment_ids go together"
                    )

                source_segment_mask = (
                    source_segment_ids[:, :, None]
                    != source_segment_ids[:, None, :]
                )

                target_segment_mask = (
                    target_segment_ids[:, :, None]
                    != target_segment_ids[:, None, :]
                )

                cross_segment_mask = (
                    target_segment_ids[:, :, None]
                    != source_segment_ids[:, None, :]
                )

            encoder_output = self.encoder(
                 source_ids,
                 key_padding_mask=source_key_padding_mask,
                 segment_mask=source_segment_mask,
            )

            decoder_output = self.decoder(
                 target_ids=target_ids,
                 memory=encoder_output,
                 tgt_key_padding_mask=target_key_padding_mask,
                 memory_key_padding_mask=source_key_padding_mask,
                 self_segment_mask=target_segment_mask,
                 cross_segment_mask=cross_segment_mask,
            )

            logits = self.llm_head(decoder_output)

            return TransformerOutput(
                 logits=logits
            )

    # ==================================================
    # Incremental decoding (KV cache)
    # ==================================================

    def init_cache(self) -> list[dict]:
        """One {"self", "cross"} pair of empty caches per decoder layer."""

        return [
            {"self": {}, "cross": {}}
            for _ in self.decoder.layers
        ]

    def encode(
            self,
            source_ids: torch.Tensor,
            source_key_padding_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:

            return self.encoder(
                source_ids,
                key_padding_mask=source_key_padding_mask,
            )

    def decode_step(
            self,
            new_target_ids: torch.Tensor,
            memory: torch.Tensor,
            cache: list[dict],
            memory_key_padding_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """
            Logits for the new tokens only, (B, new, V). Feeding the
            tokens one at a time through this reproduces forward()'s
            logits for the whole sequence, without recomputing the prefix.
            """

            decoder_output = self.decoder(
                target_ids=new_target_ids,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                cache=cache,
            )

            return self.llm_head(decoder_output)
