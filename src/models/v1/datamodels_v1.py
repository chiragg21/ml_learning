from dataclasses import dataclass
import torch

@dataclass
class LayerConfigV1:
    d_model: int
    num_heads: int
    d_ff: int
    dropout_param: float = 0.1
    # Base of RoPE's geometric frequency schedule (see modules_v1.py).
    # 10000.0 is the value used by the original RoPE paper and kept as
    # the default by nearly every model since - rarely worth changing.
    rope_theta: float = 10000.0

@dataclass
class EncoderConfigV1:
    vocab_size: int
    max_seq_len: int
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    dropout_param: float = 0.1
    rope_theta: float = 10000.0

@dataclass
class DecoderConfigV1:
    target_vocab_size: int
    max_seq_len: int
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    dropout_param: float = 0.1
    rope_theta: float = 10000.0

@dataclass
class TransformerConfigV1:
    encoder_config: EncoderConfigV1
    decoder_config: DecoderConfigV1

@dataclass
class TransformerOutput:
    logits: torch.Tensor
