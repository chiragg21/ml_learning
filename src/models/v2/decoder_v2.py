from ..template import BaseDecoder, BaseDecoderLayer
from .datamodels_v2 import DecoderConfigV2, LayerConfigV2
from ..v1.modules_v1 import RMSNorm, RotaryEmbedding, SwiGLUFeedForward
from .modules_v2 import GroupedQueryAttentionV2
import torch
from torch import nn

"""
                    Decoder Layer (v2: v1's layer with Grouped-Query Attention)
                       │
                       ▼
               ┌───────────────┐
               │   RMSNorm     │
               └───────┬───────┘
                       ▼
               ┌───────────────┐
               │ Self-Attention│
               │ (GQA + RoPE)  │
               │  tgt_mask     │
               │  tgt_padding  │
               └───────┬───────┘
                       ▼
            X1 = X + SelfAttn(RMSNorm(X))
                       │
                       ▼
               ┌───────────────┐
               │   RMSNorm     │
               └───────┬───────┘
                       ▼
               ┌───────────────┐
               │Cross-Attention│◄──── Encoder memory (already RMSNorm'd
               │ (GQA, no RoPE)│       by EncoderV2's own final_norm)
               │ memory_padding│
               └───────┬───────┘
                       ▼
            X2 = X1 + CrossAttn(RMSNorm(X1), memory)
                       │
                       ▼
               ┌───────────────┐
               │   RMSNorm     │
               └───────┬───────┘
                       ▼
               ┌───────────────┐
               │  SwiGLU FFN   │
               └───────┬───────┘
                       ▼
            X3 = X2 + FFN(RMSNorm(X2))

Why no RoPE on cross-attention: RoPE's whole point is encoding *relative*
position within a single, shared positional frame (query and key both
indexed along the same sequence). Decoder queries and encoder keys live
in two different sequences (different lengths, different positions) -
rotating each by its own sequence's position doesn't carry the same
"relative distance" meaning it does for self-attention, so most RoPE
encoder-decoder implementations (e.g. T5-style adaptations) leave
cross-attention unrotated and let self-attention (on both sides) carry
all the positional signal instead. We follow that convention here.

Mathematically, for one decoder layer with Pre-Norm:
X1 = X + SelfAttention(RMSNorm(X), causal mask)
X2 = X1 + CrossAttention(RMSNorm(X1), memory)
X3 = X2 + SwiGLU(RMSNorm(X2))

Learnable Parameters for each Decoder Layer:
d_model: D; d_ff: F; num_heads: H; num_kv_heads: G; kv_dim = G * (D/H)

 - Self-attention (no bias):  2*D*D + 2*D*kv_dim
 - Cross-attention (no bias): 2*D*D + 2*D*kv_dim
 - SwiGLU FFN (no bias):      3*D*F
 - RMSNorm x3 (gamma only):   3*D

total_per_layer = 4*D*D + 4*D*kv_dim + 3*D*F + 3*D
[4*256*256 + 4*256*64 + 3*256*1024 + 3*256
   = 262144 + 65536 + 786432 + 768 = 1,114,880]
(v1's layer at the same D/F/H: 1,311,488)
"""

class DecoderLayerV2(BaseDecoderLayer):
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
        )

        self.cross_attention = GroupedQueryAttentionV2(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout_param=config.dropout_param,
            use_rope=False,
        )

        self.ffn = SwiGLUFeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout_param=config.dropout_param,
        )

        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.norm3 = RMSNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout_param)

        self.rope = rope

    def forward(
            self,
            x,
            memory,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
    ):

        # masked self-attention
        normed = self.norm1(x)

        self_attention_output = self.self_attention(
            normed,
            normed,
            rope=self.rope,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )

        x = x + self.dropout(self_attention_output)

        # cross attention - query side pre-normed; memory (the encoder's
        # output) is already normalized by EncoderV2.final_norm, so it
        # isn't renormalized again here.
        normed = self.norm2(x)

        cross_attention_output = self.cross_attention(
            normed,
            memory,
            key_padding_mask=memory_key_padding_mask,
        )

        x = x + self.dropout(cross_attention_output)

        # SwiGLU feed-forward network
        normed = self.norm3(x)

        ffn_output = self.ffn(normed)

        x = x + self.dropout(ffn_output)

        return x


class DecoderV2(BaseDecoder):
    def __init__(
            self,
            config: DecoderConfigV2,
        ):
        super().__init__(config)
        self.target_vocab_size = config.target_vocab_size
        self.max_seq_len = config.max_seq_len
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.d_ff = config.d_ff
        self.num_layers = config.num_layers
        self.dropout_param = config.dropout_param

        # target token embeddings - no position embedding table (RoPE
        # handles position inside each layer's self-attention instead).
        # This same embedding matrix is reused as the output projection
        # by TransformerV2 (weight tying) - see transformer_v2.py.
        self.token_embeddings = nn.Embedding(
            self.target_vocab_size,
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
            DecoderLayerV2(
                LayerConfigV2(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
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
            target_ids,
            memory,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
    ):
        batch_size, target_seq_len = target_ids.shape

        if target_seq_len > self.max_seq_len:
            raise ValueError(
                "Target seq len is greater than max seq len"
            )

        x = self.token_embeddings(target_ids)
        x = self.dropout(x)

        # create causal mask if one wasn't supplied - identical
        # convention to v0 (True = masked position)
        if tgt_mask is None:
            tgt_mask = torch.triu(
                torch.ones(
                    target_seq_len,
                    target_seq_len,
                    device=target_ids.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )

        for layer in self.layers:

            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        return self.final_norm(x)
