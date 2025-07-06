from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, source_texts, target_texts, tokenizer, max_length=128):
        """
        Args:
            source_texts (list): List of source language texts.
            target_texts (list): List of target language texts.
            tokenizer (object): Tokenizer with encode_plus method.
            max_length (int): Maximum sequence length.
        """
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source = self.source_texts[idx]
        target = self.target_texts[idx]

        # Tokenize source and target texts
        source_enc = self.tokenizer.encode_plus(
            source,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        target_enc = self.tokenizer.encode_plus(
            target,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'source_ids': source_enc['input_ids'].squeeze(),
            'source_mask': source_enc['attention_mask'].squeeze(),
            'target_ids': target_enc['input_ids'].squeeze(),
            'target_mask': target_enc['attention_mask'].squeeze()
        }

