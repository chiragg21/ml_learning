import torch
from torch import nn
from ..template import BaseTransformer
from ...datamodels import TransformerOutput, TransformerConfigV0
from ..v0.encoder_v0 import EncoderV0
from ..v0.decoder_v0 import DecoderV0


class TransformerV0(BaseTransformer):
    def __init__(
            self,
            config: TransformerConfigV0
    ):
        super().__init__()  

        self.encoder = EncoderV0(config)
        self.decoder = DecoderV0(config)

        self.llm_head = nn.Linear(
            self.decoder.d_model,
            self.decoder.target_vocab_size,
        )


    def forward(
            self, 
            source_ids: torch.Tensor, 
            target_ids: torch.Tensor,
            source_key_padding_mask: torch.Tensor | None = None,
            target_key_padding_mask: torch.Tensor | None = None,
        ):

            encoder_output = self.encoder(
                 source_ids,
                 key_padding_mask = source_key_padding_mask,
            )

            decoder_output = self.decoder(
                 target_ids = target_ids,
                 memory = encoder_output,
                 tgt_key_padding_mask = target_key_padding_mask,
                 memory_key_padding_mask = source_key_padding_mask
            )

            ## Project decoder hidden states to vocabulary logits

            logits = self.llm_head(decoder_output)

            return TransformerOutput(
                 logits=logits
            )



