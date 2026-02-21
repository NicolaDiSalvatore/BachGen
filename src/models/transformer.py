"""
Summary:

1. Input Preparation:
   - Receive input tensor of token IDs (shape: batch_size x seq_len).
   - Convert tokens to embeddings of size model_dim.
   - Scale embeddings by sqrt(model_dim) to normalize variance.

2. Inject Positional Information:
   - Generate fixed sinusoidal positional encodings.
   - Add positional encodings to scaled embeddings.
   - Apply dropout for regularization.

3. Create Causal Mask:
   - Generate square mask to prevent attention to future tokens.
   - Ensures autoregressive property: token at position i attends only to tokens ≤ i.

4. Transformer Encoding:
   - Pass embeddings + positional encodings through multi-layer Transformer encoder.
   - Each encoder layer applies:
     a) Multi-head self-attention with causal masking.
     b) Feedforward neural network with activation.
     c) Residual connections and layer normalization.
   - Captures contextual dependencies across sequence positions.

5. Output Projection:
   - Apply linear layer to encoder output (shape: batch_size x seq_len x model_dim).
   - Project to output_dim (vocabulary size) logits for token prediction.

6. Output:
   - Return logits tensor with shape (batch_size, seq_len, output_dim).
   - Use for next-token prediction during training or autoregressive sampling.

"""

import torch
from torch import Tensor
from torch.nn import Dropout, Module

from src.layers.embedding import Embedding
from src.layers.feed_forward import FeedForward
from src.layers.linear import Linear
from src.layers.self_attention import SelfAttention
from src.layers.softmax import Softmax

# class Encoder(Module):
#     def __init__(self, input_dim, hidden_dim, output_dim) -> None:
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.self_attention = SelfAttention(input_dim = self.input_dim, normalization = True, output_dim = self.hidden_dim)
#         self.feed_forward = SelfAttention(input_dim = self.hidden_dim, normalization = True, output_dim = self.output_dim)
#
#     def forward(self, x: torch.Tensor) -> Tensor:
#         x = self.self_attention(x)
#         x = self.feed_forward(x)
#         return x


class Decoder(Module):
    def __init__(self, seq_len, attention_hidden_dim: int, feedforward_hidden_dim: int, num_attention_heads: int, ffn_dropout: float, attn_dropout: float, attn_proj_dropout: float) -> None:
        super().__init__()
        self.attention_hidden_dim = attention_hidden_dim
        self.seq_len = seq_len
        self.num_attention_heads = num_attention_heads
        self.self_attention = SelfAttention(input_dim=self.attention_hidden_dim, seq_len=self.seq_len, num_attention_heads=self.num_attention_heads, attn_dropout=attn_dropout, proj_dropout=attn_proj_dropout)
        self.feed_forward = FeedForward(input_dim=self.attention_hidden_dim, output_dim=self.attention_hidden_dim, hidden_dim=feedforward_hidden_dim)
        self.layer_norm1 = torch.nn.LayerNorm(attention_hidden_dim)
        self.layer_norm2 = torch.nn.LayerNorm(attention_hidden_dim)
        # self.attn_dropout = Dropout(0.1)
        self.ffn_dropout = Dropout(ffn_dropout)

    def forward(self, x: Tensor) -> Tensor:
        # print(f"Shape After Self-Attention: {y.shape}")
        x = self.layer_norm1(x + self.self_attention(x))
        # print(f"Shape After Self-Attention Normalization: {x.shape}")
        # print(f"Shape After FeedForward: {y.shape}")
        x = self.layer_norm2(x + self.ffn_dropout(self.feed_forward(x)))
        # print(f"Shape After FeedForward Normalization: {x.shape}")
        # x = self.self_attention(x)
        # x = self.feed_forward(x)
        return x


class MusicTransformer(Module):
    def __init__(self, vocab_size, seq_len, attention_hidden_dim, feedforward_hidden_dim, num_decoder_layers, num_attention_heads, embed_dropout=0.05, ffn_dropout: float = 0.1, attn_dropout: float = 0.1, attn_proj_dropout: float = 0.1) -> None:
        super().__init__()
        self.num_decoder_layers = num_decoder_layers
        self.attention_hidden_dim = attention_hidden_dim
        self.seq_len = seq_len
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.feedforward_hidden_dim = feedforward_hidden_dim
        self.embedding = Embedding(vocab_size=self.vocab_size, d_model=self.attention_hidden_dim)
        self.linear = Linear(input_dim=self.attention_hidden_dim, output_dim=self.attention_hidden_dim)
        self.embed_dropout = Dropout(embed_dropout)

        # self.encoder = Encoder(input_dim=self.d_model, output_dim=self.d_model)
        self.output_proj = Linear(input_dim=attention_hidden_dim, output_dim=vocab_size)

        self.decoders = torch.nn.ModuleList([
            Decoder(
                attention_hidden_dim=self.attention_hidden_dim,
                seq_len=self.seq_len,
                feedforward_hidden_dim=feedforward_hidden_dim,
                num_attention_heads=self.num_attention_heads,
                ffn_dropout=ffn_dropout,
                attn_dropout=attn_dropout,
                attn_proj_dropout=attn_proj_dropout
            )
            for _ in range(num_decoder_layers)
        ])
        self.softmax = Softmax()


    def forward(self, x: Tensor) -> Tensor:
        # x.shape: (batch, seq_len)
        assert x.dtype == torch.long
        assert x.min().item() >= 0, f"min token = {x.min()}"
        assert x.max().item() < self.vocab_size, f"max token = {x.max()}, vocab={self.vocab_size}"
        # print(f"Input shape: {x.shape}")
        x = self.embedding(x) # (batch, seq_len, input_dim)

        x = self.embed_dropout(x)
        # print(f"Shape after embedding: {x.shape}")
        assert not torch.isnan(x).any(), "NaNs after embedding"
        for i, decoder in enumerate(self.decoders):
            x = decoder(x)  # (batch, seq_len, input_dim)
            assert not torch.isnan(x).any(), f"NaNs after decoder in layer {i}"
        # print(f"Shape after decoders: {x.shape}")
        # x = self.linear(x)
        # x = self.softmax(x)
        x = self.output_proj(x)  # (batch, seq_len, vocab_size)
        assert not torch.isnan(x).any(), "NaNs after output projection"
        # print(f"Model output Shape: {x.shape}")
        return x
