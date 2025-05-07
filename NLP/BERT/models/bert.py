import torch
import torch.nn as nn
from models.embeddings import BERTEmbedding
from models.encoder import BERTEncoderLayer


class BERT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, d_ff=3072,
                 num_layers=12, max_len=512, dropout=0.1):
        """
        Full BERT model (Encoder only)
        Args:
            vocab_size: Size of the vocabulary
            d_model: Hidden size (768 for BERT-Base)
            num_heads: Number of attention heads (12 for BERT-Base)
            d_ff: Hidden size of FFN (3072 for BERT-Base)
            num_layers: Number of encoder layers (12 for BERT-Base)
            max_len: Max sequence length (typically 512)
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, d_model, max_len, dropout)

        self.encoder_layers = nn.ModuleList([
            BERTEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)

        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)  # project to vocabulary size
        )

        self.nsp_head = nn.Linear(d_model, 2)  # 2 classes: IsNext / NotNext

    def forward(self, token_ids, segment_ids, attention_mask=None):
        """
        Args:
            token_ids: [batch_size, seq_len]
            segment_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] or [batch_size, 1, 1, seq_len]

        Returns:
            Output tensor: [batch_size, seq_len, d_model]
        """
        # Embedding
        x = self.embedding(token_ids, segment_ids)  # [B, T, D]

        # Apply each encoder layer
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)

        x = self.layer_norm(x)

        # [CLS] token is always at position 0
        cls_token = x[:, 0]  # [batch_size, d_model]

        # MLM: apply to all tokens
        mlm_logits = self.mlm_head(x)  # [batch_size, seq_len, vocab_size]

        # NSP: apply only to CLS
        nsp_logits = self.nsp_head(cls_token)  # [batch_size, 2]

        return mlm_logits, nsp_logits


vocab_size = 30522
batch_size = 2
seq_len = 16
d_model = 768

model = BERT(vocab_size=vocab_size)

token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
segment_ids = torch.randint(0, 2, (batch_size, seq_len))
mask = (token_ids != 0).unsqueeze(1).unsqueeze(2)

mlm_logits, nsp_logits = model(token_ids, segment_ids, mask)

print("MLM shape:", mlm_logits.shape)  # [B, T, vocab_size]
print("NSP shape:", nsp_logits.shape)  # [B, 2]
