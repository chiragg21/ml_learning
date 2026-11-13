from ..template import BaseEncoder, BaseEncoderLayer
from .datamodels_v2 import EncoderConfigV2, LayerConfigV2
from ..v1.modules_v1 import RMSNorm, RotaryEmbedding, SwiGLUFeedForward
from .modules_v2 import GroupedQueryAttentionV2
import torch
import torch.nn as nn

""" Encoder Layer (v2: v1's layer with Grouped-Query Attention)

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
  (GQA, with RoPE)  │
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
v1/v2 (Pre-Norm): X1 = X + Attention(RMSNorm(X))

The layer's structure is identical to v1 - the only change is inside the
attention module, which now shares each key/value head across a group of
query heads (see GroupedQueryAttentionV2 in modules_v2.py).

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
               stack, by EncoderV2 itself; see below)

Learnable Parameters for each Encoder Layer:

d_model: D [e.g. 256]; dff: F [e.g. 1024]; num_heads: H [e.g. 8];
num_kv_heads: G [e.g. 2]; kv_dim = G * (D/H) [2 * 32 = 64]

Parameters for Attention (no biases):
 - Wq, Wo: D*D each; Wk, Wv: D*kv_dim each
 total_attn = 2*D*D + 2*D*kv_dim

Parameters for SwiGLU FFN (no biases):
 - W1, W3: D*F each; W2: F*D
 total_ffn = 3*D*F

RMSNorm x2 (gamma only, no beta):
 total_norm = 2*D

total_per_layer = 2*D*D + 2*D*kv_dim + 3*D*F + 2*D
[2*256*256 + 2*256*64 + 3*256*1024 + 2*256
   = 131072 + 32768 + 786432 + 512 = 950,784]

(For reference, v1's layer at the same D/F/H was 1,049,088 params - GQA
saves 2*D*(D - kv_dim) per attention module - and v0's was 3,152,384.
See encoder_v1.py / encoder_v0.py for their own derivations.)
"""

class EncoderLayerV2(BaseEncoderLayer):

    def __init__(
            self,
            config: LayerConfigV2,
            rope: RotaryEmbedding,
    ):
        super().__init__(config)

        self.self_attention = GroupedQueryAttentionV2(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout_param=config.dropout_param,
            use_rope=True,
            window_size=config.window_size,
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
D: d_model [256]; H: heads [8]; G: kv heads [2]; N: layers [4]

Parameters for the Encoder (v2):

Embeddings:
 - token: V*D
 - position: 0   <- this is the headline difference from v0. RoPE has
   no learned position table at all, so there's no Smax*D term here
   (v0 spent 512*512 = 262,144 params on exactly this for its default
   config). The "cost" of position information moves from a one-time
   lookup table into a fixed (non-learned) rotation applied every layer.

Stack of N layers: N * (2*D*D + 2*D*kv_dim + 3*D*F + 2*D)

Final norm (RMSNorm, applied once after the last layer - standard in
Pre-Norm architectures so the residual stream is in a sane, bounded
scale before anything downstream reads it, e.g. cross-attention or the
output head): D

Total = V*D + N*(2*D*D + 2*D*kv_dim + 3*D*F + 2*D) + D
[341*256 + 4*(950,784) + 256 = 87,296 + 3,803,136 + 256 = 3,890,688]
"""


class EncoderV2(BaseEncoder):

    def __init__(
            self,
            config: EncoderConfigV2,
        ):

        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.window_size = config.window_size
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
            EncoderLayerV2(
                LayerConfigV2(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    window_size=self.window_size,
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
