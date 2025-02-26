import torch
import torch.nn as nn

from embeddings import PositionalEncoding, TokenEmbedding
from encoder import Encoder
from decoder import Decoder


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Input Embeddings
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        # Encoder and Decoder
        self.encoder = Encoder(d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, dropout)

        # Output
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, source_tokens, target_tokens, source_mask=None, target_mask=None):
        # Encoder: Embeddings -> Positional Encoding -> Encoder -> Cross-Attention (out to decoder)
        source_embeddings = self.embedding(source_tokens)
        source_embeddings = self.positional_encoding(source_embeddings)

        encoder_outputs = self.encoder(source_embeddings, mask=source_mask)

        # Decoder: Embeddings -> Positional Encoding -> Decoder with encoder outputs
        target_embeddings = self.embedding(target_tokens)
        target_embeddings = self.positional_encoding(target_embeddings)

        decoder_output = self.decoder(target_embeddings, encoder_outputs, target_mask=target_mask)

        # Final Projection: FC Layer -> Softmax
        output_logits = self.fc_out(decoder_output)
        output = self.softmax(output_logits)

        return output


## Testing
vocab_size = 10000
d_model = 512
num_heads = 8
d_ff = 2048
seq_len = 20
batch_size = 2

# Create Transformer Model
transformer = TransformerModel(vocab_size, d_model, num_heads, d_ff)

# Dummy input (source & target sequences)
src_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))  # Random source tokens
tgt_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))  # Random target tokens

# Forward pass
output = transformer(src_tokens, tgt_tokens)

print("Output Shape:", output.shape)  # Expected: [batch_size, seq_len, vocab_size]