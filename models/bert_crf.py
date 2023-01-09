import torch
from torch import nn
from transformers import AutoModel

from datasets.ner_dataset import NerDataset
from torch.utils.data import DataLoader

LOG_INF = 10e5


class BertCRF(nn.Module):
    def __init__(self, num_labels: int, bert_name: str, dropout: float = 0.2):
        super().__init__()
        self.num_labels = num_labels
        self.start_transitions = nn.Parameter(torch.empty(num_labels))
        self.end_transitions = nn.Parameter(torch.empty(num_labels))
        self.transitions = nn.Parameter(torch.empty(num_labels, num_labels))

        self.bert = AutoModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(dropout)
        self.hidden2label = nn.Linear(self.bert.config.hidden_size, num_labels)

    def _compute_log_denominator(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        seq_len = features.shape[0]
        mask = mask.bool()

        log_score_over_all_seq = self.start_transitions + features[0]

        for i in range(1, seq_len):
            next_log_score_over_all_seq = torch.logsumexp(
                log_score_over_all_seq.unsqueeze(2)
                + self.transitions
                + features[i].unsqueeze(1),
                dim=1,
            )
            log_score_over_all_seq = torch.where(
                mask[i].unsqueeze(1),
                next_log_score_over_all_seq,
                log_score_over_all_seq,
            )
        log_score_over_all_seq += self.end_transitions
        return torch.logsumexp(log_score_over_all_seq, dim=1)

    def _compute_log_numerator(
        self, features: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        seq_len = features.shape[0]

        score_over_seq = self.start_transitions[labels[0]] + features[0, :, labels[0]]
        for i in range(1, seq_len):
            score_over_seq += (
                self.transitions[labels[i - 1], labels[i]] + features[i, :, labels[i]]
            ) * mask[i]
        seq_lens = mask.sum(dim=0) - 1
        last_tags = labels[seq_lens.long()]
        score_over_seq += self.end_transitions[last_tags]
        return score_over_seq

    def _get_bert_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.bert(input_ids, attention_mask=attention_mask)[
            "last_hidden_state"
        ]
        hidden = self.dropout(hidden)
        return self.hidden2label(hidden)

    def neg_log_likelihood(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        features = self._get_bert_features(
            input_ids=input_ids, attention_mask=attention_mask
        )

        features = torch.swapaxes(features, 0, 1)
        attention_mask = torch.swapaxes(attention_mask, 0, 1)
        labels = torch.swapaxes(labels, 0, 1)

        log_numerator = self._compute_log_numerator(
            features=features, labels=labels, mask=attention_mask
        )
        log_denominator = self._compute_log_denominator(
            features=features, mask=attention_mask
        )

        return torch.mean(log_denominator - log_numerator)

    def _viterbi_decode(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        seq_len, bs, _ = features.shape
        mask = mask.bool()

        log_score_over_all_seq = self.start_transitions + features[0]

        backpointers = torch.empty_like(features)

        for i in range(1, seq_len):
            next_log_score_over_all_seq = log_score_over_all_seq.unsqueeze(2) + self.transitions

            next_log_score_over_all_seq, indices = next_log_score_over_all_seq.max(dim=1)

            next_log_score_over_all_seq = next_log_score_over_all_seq + features[i]

            log_score_over_all_seq = torch.where(
                mask[i].unsqueeze(1),
                next_log_score_over_all_seq,
                log_score_over_all_seq,
            )
            backpointers[i] = indices

        backpointers = backpointers[1:].int()

        log_score_over_all_seq += self.end_transitions
        seq_lens = mask.sum(dim=0) - 1

        sequences = torch.zeros_like(features)

        path_scores = []
        best_paths = []
        for seq_ind in range(bs):
            best_label_id = torch.argmax(log_score_over_all_seq[seq_ind]).item()
            best_path = [best_label_id]

            for backpointer in backpointers[: seq_lens[seq_ind]]:
                print(backpointer[seq_ind][best_path[-1]])
                best_path.append(backpointer[seq_ind][best_path[-1]].item())

            best_path.reverse()
            best_paths.append(best_path)
            path_scores.append(log_score_over_all_seq[seq_ind][best_label_id].item())

        return path_scores, best_paths

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        features = self._get_bert_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        features = torch.swapaxes(features, 0, 1)
        mask = torch.swapaxes(attention_mask, 0, 1)
        return self._viterbi_decode(features=features, mask=mask)


if __name__ == "__main__":
    dataset = NerDataset("resources/data/train/tokenized_texts.jsonl")
    loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_function)
    batch = next(iter(loader))

    model = BertCRF(17, "sberbank-ai/ruBert-base")
    print(model(batch["input_ids"], batch["attention_mask"]))
