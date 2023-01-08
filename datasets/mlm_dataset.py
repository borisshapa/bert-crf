import torch
import torch.utils.data as td
import transformers


def mlm(batch: torch.Tensor, proba: float = 0.15, mask_token_id: int = 104, special_tokens=None):
    """
    Applies masked language modeling to given batch of sequence, masks each token with given probability

    :param batch: list of sequences
    :param proba: probability to hide token
    :param mask_token_id mask token id
    :param special_tokens: list of special token ids, by default [0 - PAD, 100 – UNK, 101 – CLS, 102 – SEP, 103 - MASK]

    :return: masked token ids
    """

    if special_tokens is None:
        special_tokens = [0, 100, 101, 102, 103]

    random = torch.rand_like(batch)
    hidden = (random < proba)

    for token in special_tokens:
        hidden *= (batch != token)

    for i in range(len(batch)):
        hidden_tokens_indices = hidden[i].nonzero().tolist()
        batch[i, hidden_tokens_indices] = mask_token_id

    return batch


class MaskedLanguageModelingDataset(td.Dataset):
    def __init__(
            self,
            texts: list[str],
            tokenizer: transformers.models.bert.tokenization_bert.BertTokenizer,
            mask_proba: float = 0.15,
    ):
        self.texts = texts
        self.mask_proba = mask_proba
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        return self.texts[index]

    def __len__(self):
        return len(self.texts)

    def collate_function(self, batch):
        tokenizer = self.tokenizer
        inputs = tokenizer(batch)
        labels = inputs["input_ids"].clone()
        masked = mlm(
            inputs["input_ids"],
            proba=self.mask_proba,
            mask_token_id=self.tokenizer.mask_token_id,
        )

        return {"input_ids": masked, "attention_mask": inputs["attention_mask"], "labels": "input_ids"}
