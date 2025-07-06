import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm


def train_transformer(model, dataloader, epochs, lr, device):
    """
    Train a Transformer model.

    Args:
    - model: The Transformer model instance.
    - dataloader: PyTorch DataLoader containing training data.
    - epochs: Number of epochs.
    - lr: Learning rate.
    - device: 'cuda' or 'cpu' for model training.

    Returns:
    - Trained model.
    """

    # Move model to the correct device
    model.to(device)

    # Define Loss function (Cross Entropy with label smoothing)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)  # Ignore padding tokens

    # Optimizer (AdamW is recommended for Transformers)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)  # Reduce LR every epoch

    # Training loop
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for batch in progress_bar:
            optimizer.zero_grad()

            # Move batch data to the device
            src_ids = batch['source_ids'].to(device)
            src_mask = batch['source_mask'].to(device)
            tgt_ids = batch['target_ids'].to(device)
            tgt_mask = batch['target_mask'].to(device)

            # Expands to shape [batch_size, 1, 1, seq_len]
            src_mask = src_mask.unsqueeze(1).unsqueeze(2)
            tgt_mask = tgt_mask.unsqueeze(1).unsqueeze(2)

            # Forward pass through Transformer model
            output = model(src_ids, tgt_ids, source_mask=src_mask, target_mask=tgt_mask)

            # Shift target tokens for teacher forcing
            output = output[:, :-1, :].reshape(-1, output.shape[-1])  # Remove last token
            tgt_labels = tgt_ids[:, 1:].reshape(-1)  # Remove first token

            # Compute loss
            loss = criterion(output, tgt_labels)
            loss.backward()  # Backpropagation

            # Gradient Clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")

    print("Training Completed!")
    return model
