import json
from typing import List, Dict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class NerDataset(Dataset):
    def __init__(self, tokenized_texts_path: str):
        self.tokenized_texts = []
        with open(tokenized_texts_path, "r") as tokenized_texts_file:
            for line in tokenized_texts_file:
                self.tokenized_texts.append(json.loads(line))

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, index) -> Dict[str, List]:
        return self.tokenized_texts[index]

    def collate_function(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        labels = [torch.tensor(item["labels"]) for item in batch]
        attention_mask = [torch.ones(len(item)) for item in input_ids]

        return {
            "input_ids": pad_sequence(input_ids, batch_first=True),
            "labels": pad_sequence(labels, batch_first=True),
            "attention_mask": pad_sequence(attention_mask, batch_first=True),
        }


if __name__ == "__main__":
    dataset = NerDataset("resources/data/train/tokenized_texts.jsonl")
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/ruBert-base")
    for i in range(len(dataset)):
        print(len(dataset[i]["input_ids"]), len(dataset[i]["labels"]))
    print(tokenizer.convert_ids_to_tokens(dataset[6]["input_ids"]))
