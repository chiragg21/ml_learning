import torch
from torch import nn
from ..template import BaseTransformer
from .datamodels_v1 import TransformerOutput, TransformerConfigV1
from .encoder_v1 import EncoderV1
from .decoder_v1 import DecoderV1


class TransformerV1(BaseTransformer):
    """
    v1 summary (see notebooks/mini-llm training progress.md for the full
    version roadmap): same encoder-decoder *shape* as v0, but
      - Pre-Norm instead of Post-Norm (encoder_v1.py / decoder_v1.py)
      - RMSNorm instead of LayerNorm (modules_v1.py)
      - RoPE instead of learned absolute position embeddings (modules_v1.py)
      - SwiGLU instead of a plain GELU FFN (modules_v1.py)
      - weight tying between the decoder's input embedding and the
        output projection (below)

    The forward() signature is intentionally identical to TransformerV0's
    - same kwargs, same TransformerOutput return type - so nothing in
    Trainer/Evaluator/Predictor/Generator needs to know which version
    it's holding. Switching between v0 and v1 (or between datasets) is
    purely a config change (model.transformer.version, data.name/version
    in an ExperimentConfig) - see configs/ for examples.
    """

    def __init__(
            self,
            config: TransformerConfigV1
    ):
        super().__init__(config)

        if config.encoder_config.d_model != config.decoder_config.d_model:
            raise ValueError(
                "Encoder and decoder d_model values must match for cross-attention."
            )

        self.encoder = EncoderV1(config.encoder_config)
        self.decoder = DecoderV1(config.decoder_config)

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
        A tied embedding is a real gotcha: nn.Embedding's PyTorch default
        init is Normal(mean=0, std=1) - fine for an embedding used only
        as a lookup table, but much too large once that same matrix also
        serves as a Linear output projection (which PyTorch instead
        default-initializes with std ~ 1/sqrt(in_features)). Left alone,
        logits for this d_model=256 config come out with std ~16 instead
        of ~1, which (confirmed by actually training it) blows up the
        initial loss to the 50s instead of the ~log(vocab_size) a
        randomly initialized model should start at.

        v0 never hits this because its output head is a separate,
        default-initialized nn.Linear, not tied to anything. The fix
        here is the standard one (GPT-2/nanoGPT): explicitly initialize
        every Linear/Embedding to the same small, fixed std, so the
        tied matrix is a sane scale for both of its roles at once.
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
