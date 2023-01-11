from typing import Dict

import torch


def dict_to_device(
    dict: Dict[str, torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    for key, value in dict.items():
        dict[key] = value.to(device)
    return dict
