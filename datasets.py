import json
from typing import List, Dict

import torch
import torch.utils.data as td
from torch.nn.utils.rnn import pad_sequence


class NerDataset(td.Dataset):
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


class Subset(td.Dataset):
    def __init__(self, dataset: td.Dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


class MaskedLanguageModelingDataset(td.Dataset):
    def __init__(self, file: str, device: str = "cpu"):
        self.texts = json.load(open(file, "r"))
        self.device = device

    def __getitem__(self, index):
        return self.texts[index]

    def __len__(self):
        return len(self.texts)

    def collate_function(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        return {
            "input_ids": pad_sequence(input_ids, batch_first=True).to(self.device),
            "labels": pad_sequence(labels, batch_first=True).to(self.device),
            "attention_mask": pad_sequence(attention_mask, batch_first=True).to(self.device),
        }
