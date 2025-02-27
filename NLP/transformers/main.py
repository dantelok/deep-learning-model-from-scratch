import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.transformers import TransformerModel

from utils.datesets import CustomDataset
from utils.train import train_transformer
from utils.eval import evaluate_transformer

from config import config


# Define training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = config['vocab_size']
d_model = config['d_model']
num_heads = config['num_heads']
d_ff = config['d_ff']
num_layers = config['num_layers']
dropout = config['dropout']
batch_size = config['batch_size']
epochs = config['epochs']
learning_rate = config['lr']

# Example source and target texts
source_texts = ["Hello, how are you?", "Good morning"]
target_texts = ["Bonjour, comment ça va?", "Bonjour"]

test_source_texts = ["How is the weather?", "See you later"]
test_target_texts = ["Quel temps fait-il?", "À plus tard"]

# Initialize tokenizer (e.g., using a pretrained tokenizer from Hugging Face)
tokenizer = AutoTokenizer.from_pretrained('t5-small')

# Create dataset
dataset = CustomDataset(source_texts, target_texts, tokenizer)
test_dataset = CustomDataset(test_source_texts, test_target_texts, tokenizer)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define model
transformer = TransformerModel(vocab_size, d_model, num_heads, d_ff, dropout)

# Train Transformer Model
trained_model = train_transformer(transformer, dataloader, epochs, learning_rate, device)

# Run evaluation
bleu_score = evaluate_transformer(trained_model, test_dataloader, tokenizer, device)
