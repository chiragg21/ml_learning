from abc import ABC, abstractmethod
import torch
from torch import nn    
from ..datamodels import TransformerOutput


##Encoder

class BaseEncoderLayer(nn.Module, ABC):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            dropout_param: float=0.1,
            ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_param = dropout_param

    @abstractmethod
    def forward(self):
        pass


class BaseEncoder(nn.Module, ABC):
    def __init__(
            self,
            vocab_size: int,
            max_seq_len: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            num_layers: int,
            dropout_param: float = 0.1):
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout_param = dropout_param

    @abstractmethod
    def forward(self):
        pass

## Decoder

class BaseDecoderLayer(nn.Module, ABC):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            dropout_param: float=0.1,
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_param = dropout_param

    @abstractmethod
    def forward(self):
        pass

class BaseDecoder(nn.Module, ABC):
    def __init__(
            self,
            target_vocab_size: int,
            max_seq_len: int,
            d_model: int,
            num_heads: int,
            d_ff: int, 
            num_layers: int,
            dropout_param: float = 0.1,
        ):
        self.target_vocab_size = target_vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout_param = dropout_param

    @abstractmethod
    def forward(self):
        pass


## tranformer

class BaseTransformer(nn.Module, ABC):

    @abstractmethod
    def forward(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        **kwargs,
    ) -> TransformerOutput:
        pass