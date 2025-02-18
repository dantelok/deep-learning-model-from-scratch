import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: [batch_size, num_heads, seq_len, d_k]
        mask: Optional mask for attention
        """
        pass

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        pass

    def forward(self, Q, K, V, mask=None):
        pass