from ..template import BaseDecoder, BaseDecoderLayer
from .datamodels_v0 import DecoderConfigV0, LayerConfigV0
import torch
from torch import nn

"""
                    Decoder
                       │
                       ▼
               ┌───────────────┐
               │ Self-Attention│
               │               │
               │ Q ← decoder   │
               │ K ← decoder   │
               │ V ← decoder   │
               │               │
               │ tgt_mask      │
               │ tgt_padding   │
               └───────┬───────┘
                       │
                       ▼
               ┌───────────────┐
               │Cross-Attention│◄──── Encoder memory
               │               │
               │ Q ← decoder   │
               │ K ← memory    │
               │ V ← memory    │
               │               │
               │ memory_padding│
               └───────┬───────┘
                       │
                       ▼
                      FFN

Cross Attention:
                 Encoder
                    │
                    ▼
             Encoder output E
                    │
             ┌──────┴──────┐
             │             │
             ▼             ▼
            WK            WV
             │             │
             ▼             ▼
             K             V
             ▲             ▲
             │             │
             └─────┬───────┘
                   │
                   │
Decoder ──► WQ ──► Q
                   │
                   ▼
             Q × Kᵀ
                   │
                Softmax
                   │
                   ▼
                   × V
                   │
                   ▼
            Cross-Attention
               Output

Mathematics of DecoderLayer:
X: decoder input; E: encoder output

masked self attention: 
Qs = X*Wqs + Bqs
Ks = X*Wks + Bks
Vs = X*Wvs + Bvs

    SelfAttention/SA(X) = MultiHeadAttention/MHA(Qs, Ks, Vs, causal mask)
    or SA(X) = Concat(head1, ... ,headH)*Wos + Bos;  headi = softmax(Qsi*transpose(Ksi)/sqrt(dh))*Vsi
    X1 = LayerNorm(X + Dropout(SA(X)))

cross attention:
Qc = X*Wqc + Bqc
Kc = E*Wkc + Bkc
Vc = E*Wvc + Bvc
    CrossAttention/CA(X, E) = MultiHeadAttention/MHA(Qc, Kc, Vc)
    or CA(X) = Concat(head1, ... ,headH)*Woc + Boc;  headi = softmax(Qci*transpose(Kci)/sqrt(dh))*Vci
    X2 = LayerNorm(X1 + Dropout(CA(X1, E)))
    
Feed Forward network:
    FNN(X2) = GELU(X2*W1 + b1)*W2 + b2
    X3 = LayerNorm(X2 + Dropout(FNN(X2)))
"""

class DecoderLayerV0(BaseDecoderLayer):
    def __init__(
            self,
            config: LayerConfigV0,
    ):
        super().__init__(config)
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_ff = config.d_ff
        self.dropout_param = config.dropout_param

        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout_param,
            batch_first=True,
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout_param,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.d_model),
        )

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)

        self.dropout = nn.Dropout(self.dropout_param)

    def forward(
            self,
            x,
            memory,
            tgt_mask = None,
            tgt_key_padding_mask = None,
            memory_key_padding_mask = None
    ):

        #masked self-attention
        self_attention_output,_ = self.self_attention(
            x,
            x,
            x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )

        x = self.norm1(
            x+self.dropout(self_attention_output)
        )

        # cross attention
        cross_attention_output, _ = self.cross_attention(
            x, 
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask
        )
        x = self.norm2(
            x+self.dropout(cross_attention_output)
        )

        #Feed forward network
        ffn_output = self.ffn(x)

        x = self.norm3(
            x+self.dropout(ffn_output)
        )

        return x


class DecoderV0(BaseDecoder):
    def __init__(
            self,
            config: DecoderConfigV0,
        ):
        super().__init__(config)
        self.target_vocab_size = config.target_vocab_size
        self.max_seq_len = config.max_seq_len
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_ff = config.d_ff
        self.num_layers = config.num_layers
        self.dropout_param = config.dropout_param

        #target token embeddings
        self.token_embeddings = nn.Embedding(
            self.target_vocab_size,
            self.d_model
        )

        #target positional embeddings
        self.position_embeddings = nn.Embedding(
            self.max_seq_len,
            self.d_model
        )

        self.dropout = nn.Dropout(self.dropout_param)

        #stack of decoder layers

        self.layers = nn.ModuleList([
            DecoderLayerV0(
                LayerConfigV0(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    dropout_param=self.dropout_param,
                )
            )
            for _ in range(self.num_layers)
        ])


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

        #token embeddings
        x = self.token_embeddings(target_ids)

        #positional embeddings
        positions = torch.arange(
            target_seq_len,
            device=target_ids.device,
        )

        x = x + self.position_embeddings(positions)

        x = self.dropout(x)

        #create causal mask if one wasn't supplied
        if tgt_mask is None:
            tgt_mask = torch.triu(
                torch.ones(
                    target_seq_len,
                    target_seq_len,
                    device=target_ids.device,
                    dtype=torch.bool,
                ),
                diagonal=1
            )

        #decoder layers
        for layer in self.layers:

            x = layer(
                x, 
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        return x
