import torch
import torch.nn as nn

from attention import MultiHeadAttention
from feedforward import FeedForward


class Encoder(nn.Module):
    def __init__(self, d_model, nums_head, d_ff, dropout=0.1):
        super().__init__()

        print(f"d_model = {d_model}; d_ff = {d_ff}; nums_head = {nums_head}")

        # Multi-head Attention
        self.attention = MultiHeadAttention(d_model, nums_head, dropout)

        # Layer Norm
        self.layer_norm = nn.LayerNorm(d_model)

        # Feed Forward
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # return: output, attention_weights
        attention_output, _ = self.attention(x, x, x, mask=mask)

        # Add & Norm (Residual Connection)
        x = self.layer_norm(x + self.dropout(attention_output))

        # Feed Forward
        x = self.ffn(x)

        # Add & Norm (Residual Connection)
        x = self.layer_norm(x + self.dropout(attention_output))

        return x


## Test
d_model = 512
num_heads = 8
d_ff = 2048
seq_len = 10
batch_size = 2

# Create an Encoder Block
encoder = Encoder(d_model, num_heads, d_ff)

# Dummy input tensor
x = torch.rand(batch_size, seq_len, d_model)

# Forward pass
output = encoder(x)

print("Input Shape:", x.shape)  # Expected: [batch_size, seq_len, d_model]
print("Output Shape:", output.shape)  # Expected: [batch_size, seq_len, d_model]
