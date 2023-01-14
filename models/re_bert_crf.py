from typing import Dict

import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.embeddings_and_relations_dataset import EmbeddingsAndRelationsDataset


class ReBertCrf(nn.Module):
    def __init__(
        self,
        num_re_tags: int,
        hidden_size: int,
        dropout: float,
        entity_tag_to_id: Dict[str, int],
    ):
        super().__init__()

        self.arg1_linear = self.__get_dropour_relu_linear(hidden_size, hidden_size, dropout)
        self.arg2_linear = self.__get_dropour_relu_linear(hidden_size, hidden_size, dropout)
        self.all_seq_linear = self.__get_dropour_relu_linear(hidden_size, hidden_size, dropout)

        self.relation_classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(3 * hidden_size, num_re_tags))

        self.tag_embeddings = nn.Embedding(num_embeddings=len(entity_tag_to_id), embedding_dim=hidden_size)

    def __get_dropour_relu_linear(self, from_dim: int, to_dim: int, dropout: float):
        return nn.Sequential(nn.Dropout(dropout), nn.ReLU(), nn.Linear(from_dim, to_dim))

    def forward(
        self,
        seq_embedding: torch.Tensor,
        entities_embeddings: torch.Tensor,
        entities_tags: torch.Tensor,
    ):
        bs, seq_len, emb_size = entities_embeddings.shape

        entities_embeddings_as_arg1 = self.arg1_linear(entities_embeddings)
        entities_embeddings_as_arg2 = self.arg2_linear(entities_embeddings)
        tag_embeddings = self.tag_embeddings(entities_tags)

        entities_embeddings_as_arg1 += tag_embeddings
        entities_embeddings_as_arg2 += tag_embeddings

        seq_embedding_matrix = seq_embedding.unsqueeze(1).unsqueeze(1).repeat(1, seq_len, seq_len, 1)
        grid_arg1_ind, grid_arg2_ind = torch.meshgrid(torch.arange(seq_len), torch.arange(seq_len))
        concatenated_embeddings = torch.cat(
            (
                entities_embeddings[:, grid_arg1_ind, :],
                entities_embeddings[:, grid_arg2_ind, :],
                seq_embedding_matrix,
            ),
            dim=-1,
        )
        predictions = self.relation_classifier(concatenated_embeddings)
        return predictions
