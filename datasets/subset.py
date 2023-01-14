import torch.utils.data as td


class Subset(td.Dataset):
    def __init__(self, dataset: td.Dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


__all__ = ["Subset"]
