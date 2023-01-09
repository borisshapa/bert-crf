from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class Annotation:
    id: str
    tag: str
    start_pos: int
    end_pos: int
    phrase: str


def dict_to_device(
    dict: Dict[str, torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    for key, value in dict.items():
        dict[key] = value.to(device)
    return dict
