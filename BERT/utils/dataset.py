import torch
from torch.utils.data import Dataset


class CustomBERTDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data: List of dictionaries containing pre-tokenized and pre-masked examples.
                  Each item should contain:
                    - 'input_ids': [seq_len]
                    - 'segment_ids': [seq_len]
                    - 'attention_mask': [seq_len]
                    - 'mlm_labels': [seq_len] (set non-masked positions to -100)
                    - 'nsp_labels': single int (0 or 1)
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        return {
            "input_ids": example['input_ids'].detach().clone(),
            "segment_ids": example['segment_ids'].detach().clone(),
            "attention_mask": example['attention_mask'].detach().clone(),
            "mlm_labels": example['mlm_labels'].detach().clone(),
            "nsp_labels": example['nsp_labels'].detach().clone(),
        }
