import typing as tp

import torch
import torch.utils.data as td
from torch.nn.utils.rnn import pad_sequence

from re_utils.common import load_jsonl


class JsonlDataset(td.Dataset):
    def __init__(self, dataset_path: str):
        self.texts = load_jsonl(dataset_path)

    def __getitem__(self, index) -> tp.Dict[str, tp.List]:
        return self.texts[index]

    def __len__(self):
        return len(self.texts)

    @staticmethod
    def collate_function(batch: tp.List[tp.Dict[str, torch.Tensor]]) -> tp.Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        labels = [torch.tensor(item["labels"]) for item in batch]
        attention_mask = [torch.ones(len(item)) for item in input_ids]

        return {
            "input_ids": pad_sequence(input_ids, batch_first=True),
            "labels": pad_sequence(labels, batch_first=True),
            "attention_mask": pad_sequence(attention_mask, batch_first=True),
        }


__all__ = ["JsonlDataset"]
