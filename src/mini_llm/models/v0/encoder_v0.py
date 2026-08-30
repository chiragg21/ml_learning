from ..template import Encoder, EncoderLayer
import torch
import torch.nn as nn

""" Encoder Layer
Input X
shape = (B,S,D)
       │
       │
       ├───────────────┐
       │               │
       ▼               │
     Q=XWQ             │
     K=XWK             │
     V=XWV             │
       │               │
       ▼               │
   Multi-Head          │
    Attention          │
       │               │
       ▼               │
 Attention(X)          │
       │               │
       └───────┬───────┘
               ▼
       X + Attention(X)
               │
               ▼
           LayerNorm
               │
               ▼
              X₁
               │
               ▼
       FFN(X₁)
               │
               │
       ┌───────┴────────┐
       │                │
       │                ▼
       │          X₁ + FFN(X₁)
       │                │
       │                ▼
       │            LayerNorm
       │                │
       └────────────────┘
                        ▼
                      Output

Mathematically, 
Q = X*Wq+bq; K = X*Wk+bk; V = X*Wv+bv
Head_i(X) = Attention(X) = Softmax((Q*transpose(K))/sqrt(dk))*V

MultiHeadAttention(X) = Concat(head_1, head_2, .....,  head_i)*Wo + bo

X1 = LayerNorm(X + MultiHeadAttention(X))
FFN(X1) = GELU(X1*W1+b1)W2 + b2
Output = LayerNorm(X1 + FFN(X1))

Learnable Parameters for each Encoder Layer:

d_model: D [512]; dff: F [2048]; num_heads: H [8]

Parameters for Attention: 
 - Wq, Wk, Wv, Wo: D*D
 - bq, bk, bv, bo: D

Parameters for FNN:
 - W1: F*D; b1: F
 - W2: D*F; b2: D 

LayerNorm 1 and 2
 - gamma: D; beta: D

total = 4*D*D + 2*D*F + 9*D + F 
[4*512*512+2*512*2048+9*512+2048 = 3,152,384]
"""

class EncoderLayerV0(EncoderLayer, nn.Module):

    def __init__(
            self
    ):
        super.__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.d_ff,
            dropout=self.dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.d_model)   
        )

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

        self.dropout = nn.Dropout(self.dropout_param)

    def forward(self, x, key_padding_mask = True):

        attention_output = self.self_attention(
            x,
            x, 
            x,
            key_padding_mask=key_padding_mask,
        )

        x = self.norm1(
            x + self.dropout(attention_output)
        )

        ffn_output = self.ffn(x)

        x = self.norm2(
            x + self.dropout(ffn_output)
        )

        return x

"""
Vocab Size: V [30,000]; Smax: Maximum sequence length [512]; D: d_model [512];
H: number of attention heads [8]; N: number of encoder layers [6]

Parameter for Encoders:

Embeddings:
 - token estimation: V*D
 - position estimation: Smax*D

Stack of N MultiHeadAttention:
 - N*(4*D*D + 2*D*F + 9*D + F)

Total Parameters = V*D + Smax*D + N*(4*D*D + 2*D*F + 9*D + F) 
[30,000*512 + 512*512 + 6*(3,152,384) = 34,536,448 ~ 34.5M]
"""


class EncoderV0(Encoder, nn.Module):

    def __init__(
            self
        ):

        super().__init__()

        self.token_embedding = nn.Embedding(
            self.vocab_size,
            self.d_model,
        )

        self.position_embedding = nn.Embedding(
            self.max_seq_len,
            self.d_model,
        )

        self.dropout = nn.Dropout(self.dropout_param)

        self.layers = nn.ModuleList([
            EncoderLayerV0(
                d_model = self.d_model,
                num_heads = self.num_heads,
                d_ff = self.d_ff,
                dropout_param = self.dropout_param,
            )
            for _ in range(self.num_layers)
        ])

    def forward(
            self,
            tokens,
            key_padding_mask = None
        ):

        batch_size, seq_len = tokens.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                "Sequence Length exceeds max_seq_len"
            )

        #token embedding
        x = self.token_embedding(tokens)

        #position embedding
        positions = torch.arange(
            seq_len,
            device=tokens.device,
        )

        x = x + self.position_embedding(positions)

        # dropout
        x = self.dropout(x)

        #encoder layers
        for layer in self.layers:
            x = layer(
                x,
                key_padding_mask = key_padding_mask,
            )
        
        return x