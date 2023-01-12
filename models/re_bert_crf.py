from typing import Dict

import torch
from torch import nn


class ReBertCrf(nn.Module):
    def __init__(
        self,
        num_re_tags: int,
        dropout: float
    ):
        super().__init__()

        hidden_size = self.bert_crf.bert.config.hidden_size
        self.arg1_linear = self.__get_dropour_relu_linear(
            hidden_size, hidden_size, dropout
        )
        self.arg2_linear = self.__get_dropour_relu_linear(
            hidden_size, hidden_size, dropout
        )
        self.all_seq_linear = self.__get_dropour_relu_linear(
            hidden_size, hidden_size, dropout
        )

        self.relation_classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(3 * hidden_size, num_re_tags)
        )

    def __get_dropour_relu_linear(self, from_dim: int, to_dim: int, dropout: float):
        return nn.Sequential(
            nn.Dropout(dropout), nn.ReLU(), nn.Linear(from_dim, to_dim)
        )

    def forward(self, input: Dict[str, torch.Tensor]):
        pass