from __future__ import annotations

from dataclasses import dataclass
import torch

@dataclass
class LayerConfigV2:
    d_model: int
    num_heads: int
    d_ff: int
    num_kv_heads: int | None = None
    dropout_param: float = 0.1
    rope_theta: float = 10000.0

@dataclass
class EncoderConfigV2:
    vocab_size: int
    max_seq_len: int
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    # Number of distinct key/value heads (see GroupedQueryAttentionV2).
    # None means one per query head, i.e. ordinary multi-head attention;
    # 1 is multi-query attention; anything in between is GQA proper.
    num_kv_heads: int | None = None
    dropout_param: float = 0.1
    rope_theta: float = 10000.0

@dataclass
class DecoderConfigV2:
    target_vocab_size: int
    max_seq_len: int
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    num_kv_heads: int | None = None
    dropout_param: float = 0.1
    rope_theta: float = 10000.0

@dataclass
class TransformerConfigV2:
    encoder_config: EncoderConfigV2
    decoder_config: DecoderConfigV2

@dataclass
class TransformerOutput:
    logits: torch.Tensor
