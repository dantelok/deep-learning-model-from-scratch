import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens):
        # Scale by sqrt(d_model)
        return self.embedding(tokens) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix (max_len, d_model)
        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        # sin wave for even indices
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        # cos wave for odd indices
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        positional_encoding = positional_encoding.unsqueeze(0)  # [1, max_len, d_model]

        # Store as a non-trainable buffer
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, tokens):
        # tokens shape: [batch_size, seq_len, d_model]
        tokens = tokens + self.positional_encoding[:, :tokens.size(1), :]

        return self.dropout(tokens)


# Testing
vocab_size = 10000
d_model = 512
max_len = 100
seq_len = 20
batch_size = 2

# Initialize modules
token_embedding = TokenEmbedding(vocab_size, d_model)
positional_encoding = PositionalEncoding(d_model, max_len)

# Dummy input (batch of token indices)
x = torch.randint(0, vocab_size, (batch_size, seq_len))  # Random tokens

# Forward pass
embeddings = token_embedding(x)
pos_encoded = positional_encoding(embeddings)

print("Token Embedding Shape:", embeddings.shape)  # Expected: [batch_size, seq_len, d_model]
print("Positional Encoding Shape:", pos_encoded.shape)  # Expected: [batch_size, seq_len, d_model]