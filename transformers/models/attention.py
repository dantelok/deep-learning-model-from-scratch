import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: [batch_size, num_heads, seq_len, d_k]
        mask: Optional mask for attention
        """
        d_k = Q.shape[-1]

        # Compute QK^T
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        print(f"Attention Scores Shape: {attention_scores.shape}")

        # Check Mask Shape
        if mask is not None:
            print(f"Mask Shape: {mask.shape}")
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply dropout if needed
        attention_weights = self.dropout(attention_weights)

        # SV^T
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Get dimensions
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Initialize weights
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Scaled Dot-Product Attention module
        self.attention = ScaledDotProductAttention(dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # Get batch_size
        batch_size = Q.shape[0]

        # Project multi-head inputs
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # Reshape to split head
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, d_k] -> [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Calculate Scaled Dot-Product Attention
        attention_output, attention_weights = self.attention(Q, K, V, mask)

        # Reshape back to [batch_size, seq_len, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear layer
        output = self.W_o(attention_output)

        return output, attention_weights



# Test
# Define model parameters
d_model = 512
num_heads = 8
seq_len = 10
batch_size = 2

# Create Multi-Head Attention layer
mha = MultiHeadAttention(d_model, num_heads)

# Dummy input tensors (batch_size, seq_len, d_model)
Q = torch.rand(batch_size, seq_len, d_model)
K = torch.rand(batch_size, seq_len, d_model)
V = torch.rand(batch_size, seq_len, d_model)

# Forward pass
output, attn_weights = mha(Q, K, V)

print("Output Shape:", output.shape)  # Expected: [batch_size, seq_len, d_model]
print("Attention Weights Shape:", attn_weights.shape)  # Expected: [batch_size, num_heads, seq_len, seq_len]