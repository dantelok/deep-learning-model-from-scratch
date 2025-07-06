import torch
import torch.nn as nn

from models.attention import MultiHeadAttention
from models.feedforward import FeedForward


class Decoder(nn.Module):
    def __init__(self, d_model, nums_head, d_ff, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, nums_head, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target, encoder_outputs, target_mask=None, target_key_padding_mask=None):
        # Masked Multi-head Attention
        masked_attention_output, _ = self.attention(target, target, target, mask=target_mask)

        # Add & Norm
        target = self.layer_norm(target + self.dropout(masked_attention_output))

        # Multi-head Attention with encoder output (Cross Attention)
        # Query is usually from decoder, while Key and Value are from encoder
        encoder_decoder_outputs, _ = self.attention(target, encoder_outputs, encoder_outputs, mask=target_key_padding_mask)

        # Add & Norm
        target = self.layer_norm(target + self.dropout(encoder_decoder_outputs))

        # Feed Forward
        target = self.ffn(target)

        # Add & Norm
        target = self.layer_norm(target + self.dropout(encoder_decoder_outputs))

        return target

# Define parameters
d_model = 512
num_heads = 8
d_ff = 2048
src_seq_len = 10
tgt_seq_len = 7
batch_size = 2

# Create a Decoder Block
decoder_block = Decoder(d_model, num_heads, d_ff)

# Dummy input tensors
tgt = torch.rand(batch_size, tgt_seq_len, d_model)
enc_output = torch.rand(batch_size, src_seq_len, d_model)

# Forward pass
output = decoder_block(tgt, enc_output)

print("Input Shape (tgt):", tgt.shape)  # Expected: [batch_size, tgt_seq_len, d_model]
print("Output Shape:", output.shape)  # Expected: [batch_size, tgt_seq_len, d_model]