import json

from torch.utils.data import IterableDataset

from re_utils.common import load_jsonl


class GroundTruthRelationsDataset(IterableDataset):
    def __init__(self, relations_path: str):
        self.relations = load_jsonl(relations_path)

    def __len__(self):
        return len(self.relations)

    def __getitem__(self, index):
        return self.relations[index]["relations"]
