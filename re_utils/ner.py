from typing import List, Dict, Union

import torch


def get_tags_with_positions(labels: List[int], id2label: Dict[int, str]) -> List[Dict[str, Union[str, List[int]]]]:
    tags_pos = []
    ind = 0
    while ind < len(labels):
        if id2label[labels[ind]].startswith("B"):
            tag = id2label[labels[ind]].split("-")[1]
            start_pos = ind
            ind += 1
            while ind < len(labels) and id2label[labels[ind]].startswith("I"):
                ind += 1
            end_pos = ind
            tags_pos.append({"tag": tag, "pos": [start_pos, end_pos]})
        else:
            ind += 1
    return tags_pos


def get_mean_vector_from_segment(embeddings: torch.Tensor, start_pos: int, end_pos: int) -> torch.Tensor:
    return embeddings[start_pos:end_pos].mean(dim=0)
