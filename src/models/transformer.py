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

This exact sequence ensures the model learns temporal dependencies and generates coherent,
high-quality Bach-style music by predicting each token conditioned on past tokens only.
"""


import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding for batch-first format.
    Injects information about token position in the sequence.
    """
    def __init__(self, model_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, model_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, model_dim]
        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MusicTransformer(nn.Module):
    """
    Encoder-only Transformer model for music generation.
    Uses batch-first format and causal masking.
    """
    def __init__(self, input_dim, model_dim, n_heads, n_layers, output_dim, dropout=0.1):
        super(MusicTransformer, self).__init__()
        self.model_dim = model_dim

        self.embedding = nn.Embedding(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True  # key change
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        """
        Args:
            src: Input tensor of token indices, shape [batch_size, seq_len]
        Returns:
            Logits of shape [batch_size, seq_len, output_dim]
        """
        src = self.embedding(src) * math.sqrt(self.model_dim)  # [batch_size, seq_len, model_dim]
        src = self.pos_encoder(src)  # Add positional encodings

        seq_len = src.size(1)
        mask = self.generate_causal_mask(seq_len).to(src.device)  # [seq_len, seq_len]

        output = self.transformer_encoder(src, mask=mask)  # [batch_size, seq_len, model_dim]
        return self.fc_out(output)  # Project to [batch_size, seq_len, output_dim]

    def generate_causal_mask(self, seq_len):
        """
        Create a causal mask to prevent attention to future tokens.
        Output shape: [seq_len, seq_len]
        """
        mask = torch.full((seq_len, seq_len), float(0.0))
        mask = mask.masked_fill(torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(), float('-inf'))
        return mask