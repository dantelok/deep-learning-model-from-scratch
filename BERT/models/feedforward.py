import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Position-wise FeedForward Layer used in BERT
        Args:
            d_model: Hidden size of BERT (e.g., 768)
            d_ff: Intermediate size (e.g., 3072 in BERT-Base)
            dropout: Dropout rate
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()  # BERT uses GELU, not ReLU
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.final_dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.final_dropout(x)
        return x


if __name__ == '__main__':
    # Config
    batch_size = 2
    seq_len = 10
    d_model = 768
    d_ff = 3072
    dropout = 0.1

    # Dummy input
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    # Instantiation
    ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    # Forward pass
    output = ff(x)

    # Check output shape == input shape
    print("Input shape:", x.shape)        # [2, 10, 768]
    print("Output shape:", output.shape)  # [2, 10, 768]

    # Check if gradients flow
    output.mean().backward()
    print("Gradient check passed:", x.grad is not None)
