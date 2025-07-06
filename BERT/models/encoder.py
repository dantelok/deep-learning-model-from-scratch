import torch.nn as nn
from models.attention import MultiHeadAttention
from models.feedforward import FeedForward


class BERTEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        A single BERT Encoder layer = Attention + FFN
        Args:
            d_model: hidden size of the model (e.g. 768)
            num_heads: number of attention heads (e.g. 12)
            d_ff: feedforward hidden size (e.g. 3072)
            dropout: dropout rate (typically 0.1)
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Mask tensor [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # 1. Multi-Head Self-Attention + Residual + LayerNorm
        attn_output, _ = self.attention(x, x, x, mask=attention_mask)
        x = self.attn_norm(x + self.attn_dropout(attn_output))

        # 2. FeedForward Network + Residual + LayerNorm
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + self.ffn_dropout(ffn_output))

        return x
