import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_bert(model, dataloader, epochs, lr, device):
    model.to(device)
    model.train()

    # MLM: Ignore non-masked tokens using -100
    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    nsp_loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    for epoch in range(epochs):
        total_mlm_loss = 0.0
        total_nsp_loss = 0.0
        total_nsp_correct = 0
        total_nsp_examples = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mlm_labels = batch['mlm_labels'].to(device)      # shape: [B, T], -100 for unmasked
            nsp_labels = batch['nsp_labels'].to(device)      # shape: [B]

            mlm_logits, nsp_logits = model(input_ids, segment_ids, attention_mask)

            # MLM loss: [B, T, V] vs [B, T]
            mlm_loss = mlm_loss_fn(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))

            # NSP loss: [B, 2] vs [B]
            nsp_loss = nsp_loss_fn(nsp_logits, nsp_labels)

            loss = mlm_loss + nsp_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accuracy for NSP
            nsp_preds = torch.argmax(nsp_logits, dim=-1)
            correct = (nsp_preds == nsp_labels).sum().item()

            total_mlm_loss += mlm_loss.item()
            total_nsp_loss += nsp_loss.item()
            total_nsp_correct += correct
            total_nsp_examples += nsp_labels.size(0)

            progress_bar.set_postfix({
                'mlm_loss': f"{mlm_loss.item():.4f}",
                'nsp_acc': f"{(correct / nsp_labels.size(0)) * 100:.2f}%",
            })

        avg_mlm_loss = total_mlm_loss / len(dataloader)
        avg_nsp_loss = total_nsp_loss / len(dataloader)
        nsp_acc = total_nsp_correct / total_nsp_examples

        print(f"Epoch {epoch+1}: MLM Loss={avg_mlm_loss:.4f}, NSP Loss={avg_nsp_loss:.4f}, NSP Acc={nsp_acc*100:.2f}%")

        scheduler.step()

    print("Training completed.")
    return model