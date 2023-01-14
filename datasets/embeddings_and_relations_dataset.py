import json

import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from collections import defaultdict
from typing import List, Dict


class EmbeddingsAndRelationsDataset(IterableDataset):
    FLOAT_KEYS = ["seq_embedding", "entities_embeddings"]
    INT_KEYS = ["relation_matrix", "entities_tags"]

    def __init__(self, relation_training_data_path):
        self.relation_training_data_path = relation_training_data_path

    def line_mapper(self, line):
        item = json.loads(line)
        for key in self.FLOAT_KEYS:
            item[key] = torch.tensor(item[key], dtype=torch.float)
        for key in self.INT_KEYS:
            item[key] = torch.tensor(item[key], dtype=torch.long)
        return item

    def __iter__(self):
        file_iter = open(self.relation_training_data_path, "r")
        mapped_iter = map(self.line_mapper, file_iter)
        return mapped_iter

    def collate_function(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        key2list = defaultdict(list)
        for item in batch:
            for key, value in item.items():
                key2list[key].append(value)

        for i, item in enumerate(key2list["entities_embeddings"]):
            if item.shape[0] == 0:
                key2list["entities_embeddings"][i] = torch.zeros(1, 768)
        for i, item in enumerate(key2list["entities_tags"]):
            if item.shape[0] == 0:
                key2list["entities_tags"][i] = torch.tensor([0])

        max_matrix_shape = max(
            key2list["relation_matrix"], key=lambda x: x.shape[0]
        ).shape[0]
        for i, matrix in enumerate(key2list["relation_matrix"]):
            diff = max_matrix_shape - matrix.shape[0]
            key2list["relation_matrix"][i] = (
                torch.empty(max_matrix_shape, max_matrix_shape, dtype=torch.long).fill_(-100)
                if matrix.shape[0] == 0
                else F.pad(matrix, (0, diff, 0, diff), "constant", -100)
            )

        key2tensor = {}
        for key, value in key2list.items():
            key2tensor[key] = pad_sequence(value, batch_first=True)
        return key2tensor


if __name__ == "__main__":
    dataset = EmbeddingsAndRelationsDataset(
        "resources/data/train/relation_training_data.jsonl"
    )
    data_loader = DataLoader(
        dataset, batch_size=16, collate_fn=dataset.collate_function
    )
    print(next(iter(data_loader)))
