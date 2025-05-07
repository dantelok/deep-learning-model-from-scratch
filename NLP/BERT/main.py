import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.bert import BERT

from utils.dataset import CustomBERTDataset
from utils.train import train_bert
from utils.eval import evaluate_bert

from config import config


# Define training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mask_token_id = 103  # [MASK] token ID in BERT tokenizer

vocab_size = config['vocab_size']
d_model = config['d_model']
num_heads = config['num_heads']
max_len = config['max_len']
d_ff = config['d_ff']
num_layers = config['num_layers']
dropout = config['dropout']
batch_size = config['batch_size']
epochs = config['epochs']

learning_rate = config['lr']

# Example datasets
train_data = [
    {
        'input_ids': torch.randint(0, vocab_size, (max_len,)),
        'segment_ids': torch.randint(0, 2, (max_len,)),
        'attention_mask': torch.ones(max_len),
        'mlm_labels': torch.randint(0, vocab_size, (max_len,)),  # -100 for non-masked
        'nsp_labels': torch.randint(0, 2, (1,)).squeeze()
    }
    for _ in range(64)  # Mock data for 64 examples
]

test_data = train_data[:16]  # Use a subset for test

# Create dataset
train_dataset = CustomBERTDataset(train_data)
test_dataset = CustomBERTDataset(test_data)

# Create DataLoader
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define model
BERT = BERT(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)

# Train Transformer Model
trained_model = train_bert(BERT, dataloader, epochs, learning_rate, device)

# Run evaluation
avg_mlm_loss, mlm_perplexity, nsp_accuracy = evaluate_bert(trained_model, test_dataloader, device, mask_token_id)
