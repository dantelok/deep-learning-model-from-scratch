import torch
import torch.nn.functional as F


def evaluate_bert(model, dataloader, device, mask_token_id):
    model.eval()
    total_mlm_loss, total_nsp_acc = 0, 0
    total_mlm_tokens, total_nsp_examples = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mlm_labels = batch['mlm_labels'].to(device)  # -100 for unmasked positions
            nsp_labels = batch['nsp_labels'].to(device)  # 0 or 1

            mlm_logits, nsp_logits = model(input_ids, segment_ids, attention_mask)

            # MLM loss
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                mlm_labels.view(-1),
                ignore_index=-100  # Ignores non-masked tokens
            )

            # NSP accuracy
            nsp_preds = torch.argmax(nsp_logits, dim=-1)
            correct_nsp = (nsp_preds == nsp_labels).sum().item()

            # Accumulate
            total_mlm_loss += mlm_loss.item() * (mlm_labels != -100).sum().item()
            total_nsp_acc += correct_nsp
            total_mlm_tokens += (mlm_labels != -100).sum().item()
            total_nsp_examples += input_ids.size(0)

    avg_mlm_loss = total_mlm_loss / total_mlm_tokens
    mlm_perplexity = torch.exp(torch.tensor(avg_mlm_loss))
    nsp_accuracy = total_nsp_acc / total_nsp_examples

    print(f"MLM Loss: {avg_mlm_loss:.4f} | Perplexity: {mlm_perplexity:.2f}")
    print(f"NSP Accuracy: {nsp_accuracy * 100:.2f}%")

    return avg_mlm_loss, mlm_perplexity.item(), nsp_accuracy
