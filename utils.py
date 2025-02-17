"""
### STILL IMPLEMENTING ###
Common utility functions for data loading, tokenization, etc.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd

class SimpleTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}

def load_dataset(file_path, tokenizer):
    df = pd.read_csv(file_path)
    dataset = SimpleTextDataset(df, tokenizer)
    return dataset

def create_collate_fn(tokenizer):
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]

        encoding = tokenizer(
            texts,
            truncation=True,
            padding="longest",
            max_length=128,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "texts": texts,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    return collate_fn
