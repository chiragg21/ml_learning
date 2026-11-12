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

    def forward(
            self,
            source_ids: torch.Tensor,
            target_ids: torch.Tensor,
            source_key_padding_mask: torch.Tensor | None = None,
            target_key_padding_mask: torch.Tensor | None = None,
        ):

            encoder_output = self.encoder(
                 source_ids,
                 key_padding_mask=source_key_padding_mask,
            )

            decoder_output = self.decoder(
                 target_ids=target_ids,
                 memory=encoder_output,
                 tgt_key_padding_mask=target_key_padding_mask,
                 memory_key_padding_mask=source_key_padding_mask,
            )

            logits = self.llm_head(decoder_output)

            return TransformerOutput(
                 logits=logits
            )
