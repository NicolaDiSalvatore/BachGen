"""
Multi-head self-attention layer for transformer model. The positional encoding are included here
Should control relative positions embeddings
"""
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Dropout, Module
from torch.nn.parameter import Parameter

from src.layers.linear import Linear
from src.layers.softmax import Softmax


def skew(X: Tensor) -> Tensor:
    """
    This function skew a tensor of dimension (..., L, L) as described in the section 3.4.1 of the MusicTransformer paper.
    """
    L = X.size(-2)
    X = nn.functional.pad(X, (1,0), 'constant', 0)
    X = X.reshape(*X.shape[:-2], L+1, L)
    return X[..., 1:, :]
class SelfAttention(Module):
    def __init__(self, input_dim: int, seq_len: int, num_attention_heads: int, attn_dropout: float, proj_dropout: float):
        super().__init__()

        assert input_dim % num_attention_heads == 0, "input_dim must be divisible by num_heads"

        self.input_dim = input_dim
        self.num_attention_heads = num_attention_heads
        self.dim_attention_heads = input_dim // num_attention_heads
        self.max_seq_len = seq_len

        # in general transformers the second dimension is 2*seq_len-1, but in MusicTransformer the important distances are only from -L+1 to 0 (causal masking)
        self.relative_positions_embeddings = Parameter(torch.randn(self.num_attention_heads, self.max_seq_len, self.dim_attention_heads))

        self.keys_projection = Linear(input_dim=self.input_dim, output_dim=self.input_dim, bias=False)
        self.queries_projection = Linear(input_dim=self.input_dim, output_dim=self.input_dim, bias=False)
        self.values_projection = Linear(input_dim=self.input_dim, output_dim=self.input_dim, bias=False)
        self.final_projection = Linear(input_dim=self.input_dim, output_dim=self.input_dim, bias=False)
        self.softmax = Softmax(dim=-1)
        self.attn_dropout = Dropout(p=attn_dropout)
        self.proj_dropout = Dropout(p=proj_dropout)

        # self.relative_positions_embeddings = nn.Embedding(self.num_attention_heads, seq_len, self.dim_attention_heads)  ## used for the inefficient  implementation of relative attention scores

    def attention(self, queries: Tensor, keys: Tensor, values: Tensor, relative_attention_scores: Tensor, mask: Tensor) -> Tensor:

        scores = ((torch.matmul(queries, keys.transpose(-2, -1)) + relative_attention_scores)
                  / math.sqrt(self.dim_attention_heads)) # dimension (batch_size, seq_len, seq_len)

        scores = scores.masked_fill(mask, float('-inf'))

        assert not torch.all(mask, dim=-1).any(), \
            "Fully masked attention row → NaNs guaranteed"

        scores = torch.softmax(scores, dim=-1)
        scores = self.attn_dropout(scores)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any(), "Inf in scores before softmax"
        out = torch.matmul(scores, values)
        return self.proj_dropout(out) # dimension (batch_size, seq_len, input_dim)


    def forward(self, input: Tensor) -> Tensor:

        batch_size, seq_len, input_dim = input.shape
        dim_attention_heads = input_dim//self.num_attention_heads
        assert input_dim % self.num_attention_heads == 0
        d_k = input_dim // self.num_attention_heads
        assert d_k > 0
        # assert self.max_seq_len==seq_len
        # assert self.input_dim==input_dim

        queries = self.queries_projection(input) # output of dimension (batch_size, seq_len, input_dim)
        keys = self.keys_projection(input) # output of dimension (batch_size, seq_len, input_dim)
        values = self.values_projection(input) # output of dimension (batch_size, seq_len, input_dim)

        assert not torch.isnan(self.queries_projection.weights).any(), "NaNs in Q weights"

        mask = torch.triu(torch.ones(seq_len, seq_len, device=input.device, dtype=torch.bool), diagonal=1).unsqueeze(0)

        heads = []
        queries = queries.reshape(batch_size, seq_len, self.num_attention_heads, dim_attention_heads).transpose(1, 2)
        keys = keys.reshape(batch_size, seq_len, self.num_attention_heads, dim_attention_heads).transpose(1, 2)
        values = values.reshape(batch_size, seq_len, self.num_attention_heads, dim_attention_heads).transpose(1, 2)

        for h in range(self.num_attention_heads):


            rel_embed = self.relative_positions_embeddings[h][
                  self.max_seq_len - seq_len: self.max_seq_len + seq_len - 1
                  ]

            relative_attention_scores = torch.matmul(
                queries[:, h], rel_embed.transpose(0,1)
            )

            # print("relative scores shape:", relative_attention_scores.shape)
            # print("L:", seq_len)

            relative_attention_scores = skew(relative_attention_scores)
            assert not torch.isnan(queries).any()
            assert not torch.isnan(keys).any()
            assert not torch.isnan(relative_attention_scores).any()
            assert relative_attention_scores.shape[-2:] == (seq_len, seq_len)

            head_output = self.attention(
                queries[:, h],
                keys[:, h],
                values[:, h],
                relative_attention_scores,
                mask
            )

            heads.append(head_output)

        return self.final_projection(torch.cat(heads, dim=-1))


