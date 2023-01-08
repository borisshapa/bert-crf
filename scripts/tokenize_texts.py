import glob
import json
import os.path
from argparse import ArgumentParser, Namespace

from transformers import AutoTokenizer

from re_utils.common import Annotation

NOT_A_NAMED_ENTITY = "O"
FIRST_TOKEN_TAG_PREFIX = "B"
SUBSEQUENT_TOKEN_TAG_PREFIX = "I"


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
        "--save-to",
        type=str,
        default="resources/data/train",
        help="Directory where tokenized and labeled texts are saved."
    )
    return arg_parser


def main(args: Namespace):
    os.makedirs(args.save_to, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    tags_set = set()
    tokenized_texts = []
    for file in glob.glob(f"{args.train_dir}/**/*.txt", recursive=True):
        with open(file, "r") as text_file:
            annotation_path = file.split(".")[0] + ".ann"

            ner_annotations = []
            with open(annotation_path, "r") as annotation_file:
                annotation_lines = annotation_file.readlines()
                for annotation_line in annotation_lines:
                    annotation_data = annotation_line.split()
                    if not annotation_data[0].startswith("T"):
                        continue

                    annotation = Annotation(
                        id=annotation_data[0],
                        tag=annotation_data[1],
                        start_pos=int(annotation_data[2]),
                        end_pos=int(annotation_data[3]),
                        phrase=" ".join(annotation_data[4:]),
                    )
                    tags_set.add(annotation.tag)
                    ner_annotations.append(annotation)

            pos2annotation = {
                (ann.start_pos, ann.end_pos): ann for ann in ner_annotations
            }
            words = []
            tags = []

            text = text_file.read()
            word = ""
            pos = [0, 0]
            for ind, ch in enumerate(text):
                if ch.isspace():
                    pos[1] = ind
                    if len(word) > 0:
                        annotation = pos2annotation.get(tuple(pos))
                        tag = annotation.tag if annotation else NOT_A_NAMED_ENTITY
                        words.append(word)
                        tags.append(tag)
                        word = ""
                    pos[0] = ind + 1
                    continue
                word += ch

            encoded = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
            text_labels = []

            prev_word_ind = None
            prev_token_ind = 0
            word_labels = []

            part_id = 0
            for token_ind, word_ind in enumerate(encoded.word_ids()):
                if word_ind == prev_word_ind and prev_word_ind is not None:
                    tag = tags[word_ind]
                    word_labels.append(
                        f"{SUBSEQUENT_TOKEN_TAG_PREFIX}-{tag}"
                        if tag != NOT_A_NAMED_ENTITY
                        else NOT_A_NAMED_ENTITY
                    )
                else:
                    text_labels.extend(word_labels)
                    word_labels.clear()

                    tag = tags[word_ind] if word_ind is not None else NOT_A_NAMED_ENTITY
                    word_labels.append(
                        f"{FIRST_TOKEN_TAG_PREFIX}-{tag}"
                        if tag != NOT_A_NAMED_ENTITY
                        else NOT_A_NAMED_ENTITY
                    )

                if len(text_labels) + len(word_labels) > args.max_seq_len:
                    tokenized_texts.append({
                        "input_ids": encoded["input_ids"][prev_token_ind:prev_token_ind + len(text_labels)],
                        "text_labels": text_labels
                    })
                    text_labels.clear()
                    part_id += 1

            text_labels.extend(word_labels)
            if len(text_labels) > 0:
                tokenized_texts.append({
                    "input_ids": encoded["input_ids"][prev_token_ind:-1],
                    "text_labels": text_labels
                })

    labels_set = set()
    for tag in tags_set:
        labels_set.add(f"{FIRST_TOKEN_TAG_PREFIX}-{tag}")
        labels_set.add(f"{SUBSEQUENT_TOKEN_TAG_PREFIX}-{tag}")
    labels_set.add(NOT_A_NAMED_ENTITY)

    label2id = {label: id for id, label in enumerate(labels_set)}
    id2label = {id: label for label, id in label2id.items()}

    for i in range(len(tokenized_texts)):
        tokenized_texts[i]["labels"] = [label2id[label] for label in tokenized_texts[i]["text_labels"]]

    with open(os.path.join(args.save_to, "tokenized_texts.json"), "w") as tokenized_texts_file:
        json.dump(tokenized_texts, tokenized_texts_file)
    with open(os.path.join(args.save_to, "label2id.json"), "w") as label2id_file:
        json.dump(label2id, label2id_file)
    with open(os.path.join(args.save_to, "id2label.json"), "w") as id2label_file:
        json.dump(id2label, id2label_file)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(_args)
