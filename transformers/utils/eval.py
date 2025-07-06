import torch
import torch.nn.functional as F
from tqdm import tqdm
from sacrebleu import corpus_bleu  # Install using: pip install sacrebleu


def evaluate_transformer(model, test_dataloader, tokenizer, device):
    """
    Evaluate the Transformer model on test data.

    Args:
    - model: The trained Transformer model.
    - test_dataloader: DataLoader for test set.
    - tokenizer: Tokenizer for decoding outputs.
    - device: 'cuda' or 'cpu' for inference.

    Returns:
    - BLEU Score (if applicable).
    """
    model.eval()  # Set model to evaluation mode
    references = []  # Ground-truth translations
    hypotheses = []  # Model-generated translations
    total_loss = 0.0

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            src_ids = batch['source_ids'].to(device)
            src_mask = batch['source_mask'].to(device)
            tgt_ids = batch['target_ids'].to(device)
            tgt_mask = batch['target_mask'].to(device)

            # Expands to shape [batch_size, 1, 1, seq_len]
            src_mask = src_mask.unsqueeze(1).unsqueeze(2)
            tgt_mask = tgt_mask.unsqueeze(1).unsqueeze(2)

            # Forward pass (No teacher forcing)
            output_logits = model(src_ids, tgt_ids, source_mask=src_mask, target_mask=tgt_mask)

            # Compute loss
            output_logits = output_logits[:, :-1, :].reshape(-1, output_logits.shape[-1])
            tgt_labels = tgt_ids[:, 1:].reshape(-1)
            loss = criterion(output_logits, tgt_labels)
            total_loss += loss.item()

            # Convert predicted tokens to text
            predicted_ids = torch.argmax(output_logits, dim=-1)  # Get highest probability token
            predicted_texts = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predicted_ids]
            target_texts = [tokenizer.decode(tgt, skip_special_tokens=True) for tgt in tgt_ids[:, 1:]]

            # Store results for BLEU evaluation
            hypotheses.extend(predicted_texts)
            references.extend([[ref] for ref in target_texts])  # BLEU requires nested lists

    avg_loss = total_loss / len(test_dataloader)
    bleu_score = corpus_bleu(hypotheses, references).score  # Compute BLEU score

    print(f"\nEvaluation Completed!")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"BLEU Score: {bleu_score:.2f}")

    return bleu_score