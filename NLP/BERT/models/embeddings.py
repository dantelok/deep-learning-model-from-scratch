import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens):
        return self.embedding(tokens) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    """Different from Transformer-bsed model; it's trainable"""
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, seq_len]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        return self.pos_embedding(positions)


class SegmentEmbedding(nn.Module):
    """For NSP (Next Sentence Prediction)"""
    def __init__(self, d_model):
        super().__init__()
        # 0 = sentence A, 1 = sentence B -> 2 sentences
        self.segment_embeddings = nn.Embedding(2, d_model)

    def forward(self, segment_ids):
        """
        segment_ids: [batch_size, seq_len] with values 0 or 1
        Returns: [batch_size, seq_len, d_model]
        """
        return self.segment_embeddings(segment_ids)


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, d_model)
        self.position = PositionalEmbedding(max_len, d_model)
        self.segment = SegmentEmbedding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids, segment_ids):
        token_embed = self.token(token_ids)
        position_embed = self.position(token_ids)  # uses shape only
        segment_embed = self.segment(segment_ids)

        # Sum all 3 embeddings
        x = token_embed + position_embed + segment_embed
        return self.dropout(self.layer_norm(x))

# Test
vocab_size = 1000
d_model = 768
batch_size = 2
seq_len = 10

# Dummy Input
token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))   # [batch_size, seq_len]
segment_ids = torch.randint(0, 2, (batch_size, seq_len))          # [batch_size, seq_len], values in {0, 1}

# Instantiate and Run
embedding = BERTEmbedding(vocab_size=vocab_size, d_model=d_model)
output = embedding(token_ids, segment_ids)

# Output Check
print("Input token_ids shape:", token_ids.shape)
print("Input segment_ids shape:", segment_ids.shape)
print("Output embeddings shape:", output.shape)  # Expected: [batch_size, seq_len, d_model]
