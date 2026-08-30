from dataclasses import dataclass
import torch

@dataclass
class TransformerConfigV0:
    vocab_size: int
    max_seq_len: int
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    dropout_param: float = 0.1

@dataclass
class TransformerOutput:
    logits: torch.Tensor
    