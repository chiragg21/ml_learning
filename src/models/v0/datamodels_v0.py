from dataclasses import dataclass
import torch

@dataclass
class LayerConfigV0:
    d_model: int
    num_heads: int
    d_ff: int
    dropout_param: float = 0.1

@dataclass
class EncoderConfigV0:
    vocab_size: int
    max_seq_len: int
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    dropout_param: float = 0.1

@dataclass
class DecoderConfigV0:
    target_vocab_size: int
    max_seq_len: int
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    dropout_param: float = 0.1

@dataclass
class TransformerConfigV0:
    encoder_config: EncoderConfigV0
    decoder_config: DecoderConfigV0

@dataclass
class TransformerOutput:
    logits: torch.Tensor