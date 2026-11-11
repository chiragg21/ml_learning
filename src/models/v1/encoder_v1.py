from ..template import BaseEncoder, BaseEncoderLayer
from .datamodels_v1 import EncoderConfigV1, LayerConfigV1
from .modules_v1 import MultiHeadAttentionV1, RMSNorm, RotaryEmbedding, SwiGLUFeedForward
import torch
import torch.nn as nn

""" Encoder Layer (v1: Pre-Norm + RMSNorm + RoPE + SwiGLU)

Input X
shape = (B,S,D)
   │
   ├───────────────┐
   │               │
   ▼               │
RMSNorm(X)         │
   │               │
   ▼               │
SelfAttention       │
  (with RoPE)       │
   │               │
   └───────┬───────┘
           ▼
   X + Attention(RMSNorm(X))
           │
           ├───────────────┐
           │               │
           ▼               │
      RMSNorm(X1)          │
           │               │
           ▼               │
      SwiGLU FFN            │
           │               │
           └───────┬───────┘
                   ▼
           X1 + FFN(RMSNorm(X1))
                   │
                   ▼
                 Output

v0 (Post-Norm):  X1 = LayerNorm(X + Attention(X))
v1 (Pre-Norm):   X1 = X + Attention(RMSNorm(X))

Normalizing the *input* to each sublayer, instead of its output, keeps an
unnormalized residual "stream" flowing all the way through the network's
depth. That stream is what each layer reads from and adds back into -
keeping it clean (rather than periodically renormalizing it, as Post-Norm
does) is what made much deeper transformers trainable without careful
learning-rate warmup/init tuning. Every major LLM since GPT-2 uses Pre-Norm.

Mathematically, for one encoder layer with Pre-Norm:
X1 = X + SelfAttention(RMSNorm(X))
X2 = X1 + SwiGLU(RMSNorm(X1))
Output = X2   (no final norm here - that's applied once, after the whole
               stack, by EncoderV1 itself; see below)

Learnable Parameters for each Encoder Layer:

d_model: D [e.g. 256]; dff: F [e.g. 1024]; num_heads: H [e.g. 8]

Parameters for Attention (no biases):
 - Wq, Wk, Wv, Wo: D*D each
 total_attn = 4*D*D

Parameters for SwiGLU FFN (no biases):
 - W1, W3: D*F each; W2: F*D
 total_ffn = 3*D*F

RMSNorm x2 (gamma only, no beta):
 total_norm = 2*D

total_per_layer = 4*D*D + 3*D*F + 2*D
[4*256*256 + 3*256*1024 + 2*256 = 262144 + 786432 + 512 = 1,049,088]

(For reference, v0's equivalent layer at the same D/F/H was 3,152,384
params - see encoder_v0.py's own derivation.)
"""

class EncoderLayerV1(BaseEncoderLayer):

    def __init__(
            self,
            config: LayerConfigV1,
            rope: RotaryEmbedding,
    ):
        super().__init__(config)

        self.self_attention = MultiHeadAttentionV1(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout_param=config.dropout_param,
            use_rope=True,
        )

        self.ffn = SwiGLUFeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout_param=config.dropout_param,
        )

        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout_param)

        self.rope = rope

    def forward(self, x, key_padding_mask=None):

        normed = self.norm1(x)

        attention_output = self.self_attention(
            normed,
            normed,
            rope=self.rope,
            key_padding_mask=key_padding_mask,
        )

        x = x + self.dropout(attention_output)

        normed = self.norm2(x)

        ffn_output = self.ffn(normed)

        x = x + self.dropout(ffn_output)

        return x


"""
Vocab Size: V [e.g. 341]; Smax: Maximum sequence length [e.g. 160];
D: d_model [256]; H: heads [8]; N: layers [4]

Parameters for the Encoder (v1):

Embeddings:
 - token: V*D
 - position: 0   <- this is the headline difference from v0. RoPE has
   no learned position table at all, so there's no Smax*D term here
   (v0 spent 512*512 = 262,144 params on exactly this for its default
   config). The "cost" of position information moves from a one-time
   lookup table into a fixed (non-learned) rotation applied every layer.

Stack of N layers: N * (4*D*D + 3*D*F + 2*D)

Final norm (RMSNorm, applied once after the last layer - standard in
Pre-Norm architectures so the residual stream is in a sane, bounded
scale before anything downstream reads it, e.g. cross-attention or the
output head): D

Total = V*D + N*(4*D*D + 3*D*F + 2*D) + D
[341*256 + 4*(1,049,088) + 256 = 87,296 + 4,196,352 + 256 = 4,283,904]
"""


class EncoderV1(BaseEncoder):

    def __init__(
            self,
            config: EncoderConfigV1,
        ):

        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_ff = config.d_ff
        self.num_layers = config.num_layers
        self.dropout_param = config.dropout_param

        self.token_embedding = nn.Embedding(
            self.vocab_size,
            self.d_model,
        )

        self.dropout = nn.Dropout(self.dropout_param)

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        head_dim = self.d_model // self.num_heads

        self.rope = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=self.max_seq_len,
            theta=config.rope_theta,
        )

        self.layers = nn.ModuleList([
            EncoderLayerV1(
                LayerConfigV1(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    dropout_param=self.dropout_param,
                    rope_theta=config.rope_theta,
                ),
                rope=self.rope,
            )
            for _ in range(self.num_layers)
        ])

        self.final_norm = RMSNorm(self.d_model)

    def forward(
            self,
            tokens,
            key_padding_mask=None,
        ):

        batch_size, seq_len = tokens.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                "Sequence Length exceeds max_seq_len"
            )

        # token embedding only - no position embedding added here.
        # Position is injected inside each layer's self-attention via
        # RoPE instead (see modules_v1.RotaryEmbedding).
        x = self.token_embedding(tokens)

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(
                x,
                key_padding_mask=key_padding_mask,
            )

        return self.final_norm(x)
