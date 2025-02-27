import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model
