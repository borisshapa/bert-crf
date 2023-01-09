import json

import torch.utils.data as td
from torch.nn.utils.rnn import pad_sequence


class MaskedLanguageModelingDataset(td.Dataset):
    def __init__(self, jsonl_file: str, device: str = "cpu"):
        self.masked_texts = []
        self.device = device

        with open(jsonl_file, "r") as file:
            for line in file:
                self.masked_texts.append(json.loads(line))

    def __getitem__(self, index):
        return self.masked_texts[index]

    def __len__(self):
        return len(self.masked_texts)

    def collate_function(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        return {
            "input_ids": pad_sequence(input_ids, batch_first=True).to(self.device),
            "labels": pad_sequence(labels, batch_first=True).to(self.device),
            "attention_mask": pad_sequence(attention_mask, batch_first=True).to(self.device),
        }


__all__ = [
    "MaskedLanguageModelingDataset"
]
