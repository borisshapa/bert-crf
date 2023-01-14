import json
from collections import defaultdict
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, DataLoader


class EmbeddingsAndRelationsDataset(IterableDataset):
    FLOAT_KEYS = ["seq_embedding", "entities_embeddings"]
    INT_KEYS = ["relation_matrix", "entities_tags", "entities_positions"]

    def __init__(self, re_data_path):
        self.re_data_path = re_data_path

    def line_mapper(self, line):
        item = json.loads(line)
        for key in self.FLOAT_KEYS:
            item[key] = torch.tensor(item[key], dtype=torch.float)
        for key in self.INT_KEYS:
            item[key] = torch.tensor(item[key], dtype=torch.long)
        return item

    def __iter__(self):
        file_iter = open(self.re_data_path, "r")
        mapped_iter = map(self.line_mapper, file_iter)
        return mapped_iter

    def collate_function(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        key2list = defaultdict(list)
        for item in batch:
            for key, value in item.items():
                key2list[key].append(value)

        ids = torch.tensor(key2list["id"], dtype=torch.long)
        del key2list["id"]

        key2item_shape = {"entities_embeddings": (1, 768), "entities_tags": (1), "entities_positions": (1, 2)}
        for key, item_shape in key2item_shape.items():
            for i in range(len(key2list[key])):
                if key2list[key][i].shape[0] == 0:
                    key2list[key][i] = torch.zeros(
                        item_shape, dtype=torch.float if key in self.FLOAT_KEYS else torch.int
                    )

        max_matrix_shape = max(key2list["relation_matrix"], key=lambda x: x.shape[0]).shape[0]
        for i, matrix in enumerate(key2list["relation_matrix"]):
            diff = max_matrix_shape - matrix.shape[0]
            key2list["relation_matrix"][i] = (
                torch.empty(max_matrix_shape, max_matrix_shape, dtype=torch.long).fill_(-100)
                if matrix.shape[0] == 0
                else F.pad(matrix, (0, diff, 0, diff), "constant", -100)
            )

        key2tensor = {"id": ids}
        for key, value in key2list.items():
            key2tensor[key] = pad_sequence(value, batch_first=True)
        return key2tensor
