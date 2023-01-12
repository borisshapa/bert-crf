from transformers import AutoTokenizer

from datasets.jsonl_dataset import JsonlDataset


class NerDataset(JsonlDataset):
    pass


__all__ = [
    "NerDataset"
]

if __name__ == "__main__":
    dataset = NerDataset("resources/data/train/tokenized_texts.jsonl")
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/ruBert-base")
    for i in range(len(dataset)):
        print(len(dataset[i]["input_ids"]), len(dataset[i]["labels"]))
    print(tokenizer.convert_ids_to_tokens(dataset[6]["input_ids"]))
