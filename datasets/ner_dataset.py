import json
from typing import List, Dict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class NerDataset(Dataset):
    def __init__(self, tokenized_texts_path: str):
        with open(tokenized_texts_path, "r") as tokenized_texts_file:
            self.tokenized_texts = json.load(tokenized_texts_file)

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, index):
        return self.tokenized_texts[index]

    def collate_function(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        attention_mask = [torch.ones(len(item)) for item in input_ids]

        return {
            "input_ids": pad_sequence(input_ids, batch_first=True),
            "labels": pad_sequence(labels, batch_first=True),
            "attention_mask": pad_sequence(attention_mask, batch_first=True),
        }
