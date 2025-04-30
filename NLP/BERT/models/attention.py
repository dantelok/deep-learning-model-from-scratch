import torch
import torch.nn as nn
from torch.nn.functional import F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = q.shape[-1]

        attention_score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, float('-inf'))

        attention_weight = F.softmax(attention_score, dim=-1)
        attention_weight = self.dropout(attention_weight)
        attention_output = torch.matmul(attention_weight, v)

        return attention_output, attention_weight


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Calculate Scaled Dot-Product Attention
        attn_output, attn_weights = self.attention(q, k, v)

        # Scale back to [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Final layer
        attn_output = self.w_o(attn_output)

        return attn_output, attn_weights
