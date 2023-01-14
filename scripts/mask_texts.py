import glob
import json
import os.path
import random
from argparse import ArgumentParser, Namespace

import transformers
from tqdm import tqdm
from transformers import AutoTokenizer


def configure_arg_parser():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--train-dir",
        type=str,
        default="resources/data/train",
        help="Directory where the source data is located",
    )
    arg_parser.add_argument(
        "--hf-tokenizer",
        type=str,
        default="sberbank-ai/ruBert-base",
        help="The name of the tokenizer with which to tokenize the text. "
        "This can be a tokenizer from the hf pub or a local path.",
    )
    arg_parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length in tokens.",
    )
    arg_parser.add_argument(
        "--masked-proba",
        type=float,
        default=0.15,
        help="Probability for masked language modeling. Each token will be hidden with given probability.",
    )
    arg_parser.add_argument(
        "--save-to",
        type=str,
        default="resources/data/train",
        help="Directory where tokenized and labeled texts are saved.",
    )
    return arg_parser


def main(args: Namespace):
    os.makedirs(args.save_to, exist_ok=True)

    tokenizer: transformers.BertTokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    masked_texts = []

    for file in tqdm(glob.glob(f"{args.train_dir}/**/*.txt", recursive=True)):
        with open(file, "r") as text_file:
            text = text_file.read()
            encoded = tokenizer(text, add_special_tokens=False)

            labels = []
            input_ids = []

            for token in encoded["input_ids"][: args.max_seq_len]:
                labels.append(token)

                if random.random() < args.masked_proba:
                    input_ids.append(tokenizer.mask_token_id)
                else:
                    input_ids.append(token)

            masked_texts.append({"input_ids": input_ids, "labels": labels})

    with open(os.path.join(args.save_to, "masked_texts.jsonl"), "w") as masked_text_file:
        for masked_text in masked_texts:
            assert len(masked_text["input_ids"]) == len(masked_text["labels"])
            json.dump(masked_text, masked_text_file)
            masked_text_file.write("\n")


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(_args)
