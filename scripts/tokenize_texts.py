import copy
import glob
import json
import os.path
from argparse import ArgumentParser, Namespace
from typing import Optional, Set

from nltk.tokenize import WordPunctTokenizer
from tqdm import tqdm
from transformers import AutoTokenizer

from re_utils.common import (
    NerAnnotation,
    ReAnnotation,
    lower_bound,
    upper_bound,
    binary_search,
    save_jsonl,
    save_json,
)

NOT_A_NAMED_ENTITY = "O"
FIRST_TOKEN_TAG_PREFIX = "B"
SUBSEQUENT_TOKEN_TAG_PREFIX = "I"


def configure_arg_parser():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--dir",
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
        "--label2id",
        type=str,
        default=None,
        help="json file with mapping from label name to id",
    )
    arg_parser.add_argument(
        "--retag2id",
        type=str,
        default=None,
        help="json file with mapping from relation tag to id",
    )
    return arg_parser


def get_mapping_to_id(argument: Optional[str], set: Set[str]):
    if argument is not None:
        with open(argument, "r") as label2id_file:
            label2id = json.load(label2id_file)
    else:
        label2id = {label: id for id, label in enumerate(set)}
    return label2id


def main(args: Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)

    tokenized_texts = []
    relations = []

    skipped_relations = 0

    labels_set = set()
    retags_set = set()

    text_id = 0
    for text_path in tqdm(
        glob.glob(
            f"{args.dir}/**/*.txt",
            recursive=True,
        )
    ):
        with open(text_path, "r") as text_file:
            text = text_file.read()

        annotation_path = os.path.join(
            os.path.dirname(text_path),
            os.path.basename(text_path).split(".")[0] + ".ann",
        )
        ner_annotations = []
        re_annotations = []

        with open(annotation_path, "r") as annotation_file:
            for annotation_line in annotation_file.readlines():
                annotation_data = annotation_line.split()
                annotation_id = annotation_data[0]
                if annotation_id.startswith("T"):
                    ner_annotation = NerAnnotation(
                        id=annotation_id,
                        tag=annotation_data[1],
                        start_ch_pos=int(annotation_data[2]),
                        end_ch_pos=int(annotation_data[3]),
                        phrase=" ".join(annotation_data[4:]),
                    )
                    ner_annotations.append(ner_annotation)

                    labels_set.add(f"{FIRST_TOKEN_TAG_PREFIX}-{ner_annotation.tag}")
                    labels_set.add(f"{SUBSEQUENT_TOKEN_TAG_PREFIX}-{ner_annotation.tag}")
                else:
                    annotation_id, tag, arg1, arg2 = annotation_data

                    def get_arg_name(arg: str):
                        return arg.split(":")[1]

                    arg1 = get_arg_name(arg1)
                    arg2 = get_arg_name(arg2)

                    re_annotations.append(ReAnnotation(id=annotation_id, tag=tag, arg1=arg1, arg2=arg2))
                    retags_set.add(tag)

        id2annotation = {ann.id: ann for ann in ner_annotations}
        tokenized_text_spans = list(WordPunctTokenizer().span_tokenize(text))

        for id in id2annotation.keys():
            start_ch_pos = id2annotation[id].start_ch_pos
            end_ch_pos = id2annotation[id].end_ch_pos

            start_word_pos_ind = upper_bound(tokenized_text_spans, start_ch_pos, key=lambda x: x[0])
            start_word_pos_ind -= 1
            start = tokenized_text_spans[start_word_pos_ind][0]
            if start != start_ch_pos:
                end = tokenized_text_spans[start_word_pos_ind][1]
                tokenized_text_spans[start_word_pos_ind] = (start, start_ch_pos)
                tokenized_text_spans.insert(start_word_pos_ind + 1, (start_ch_pos, end))

            end_word_pos_ind = lower_bound(tokenized_text_spans, end_ch_pos, lambda x: x[1])
            end = tokenized_text_spans[end_word_pos_ind][1]
            if tokenized_text_spans[end_word_pos_ind][1] != end_ch_pos:
                start = tokenized_text_spans[end_word_pos_ind][0]
                tokenized_text_spans[end_word_pos_ind] = (end_ch_pos, end)
                tokenized_text_spans.insert(end_word_pos_ind, (start, end_ch_pos))

        for id in id2annotation.keys():
            id2annotation[id].start_word_pos = binary_search(
                tokenized_text_spans, id2annotation[id].start_ch_pos, lambda x: x[0]
            )
            id2annotation[id].end_word_pos = binary_search(
                tokenized_text_spans, id2annotation[id].end_ch_pos, lambda x: x[1]
            )

        for annotation in id2annotation.values():
            assert annotation.start_word_pos != -1 and annotation.end_word_pos != -1

        words = [text[span[0] : span[1]] for span in tokenized_text_spans]
        encoded = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        words_ids_for_tokens = encoded.word_ids()

        for id in id2annotation.keys():
            id2annotation[id].start_token_pos = lower_bound(words_ids_for_tokens, id2annotation[id].start_word_pos)
            id2annotation[id].end_token_pos = upper_bound(words_ids_for_tokens, id2annotation[id].end_word_pos)

        text_labels = ["O"] * len(input_ids)
        for annotation in id2annotation.values():
            text_labels[annotation.start_token_pos] = f"{FIRST_TOKEN_TAG_PREFIX}-{annotation.tag}"
            for i in range(annotation.start_token_pos + 1, annotation.end_token_pos):
                text_labels[i] = f"{SUBSEQUENT_TOKEN_TAG_PREFIX}-{annotation.tag}"

        labels_set.add(NOT_A_NAMED_ENTITY)

        current_seq_ids = []
        current_seq_labels = []
        dump = {"input_ids": [], "text_labels": [], "labels": []}
        total_token_dumped = 0

        relations_count = 0
        for token_ind in range(len(input_ids)):
            is_new_word = words_ids_for_tokens[token_ind] is None or (
                token_ind > 0 and words_ids_for_tokens[token_ind] != words_ids_for_tokens[token_ind - 1]
            )

            is_not_subseq_label = text_labels[token_ind] == NOT_A_NAMED_ENTITY or text_labels[token_ind].startswith("B")

            if is_new_word and is_not_subseq_label:
                dump["input_ids"].extend(current_seq_ids.copy())
                dump["text_labels"].extend(current_seq_labels.copy())
                current_seq_ids.clear()
                current_seq_labels.clear()

            current_seq_ids.append(input_ids[token_ind])
            current_seq_labels.append(text_labels[token_ind])

            if len(current_seq_ids) + len(dump["input_ids"]) >= args.max_seq_len:
                dump_relations = {"id": text_id, "relations": []}
                for re_annotation in re_annotations:
                    arg1 = re_annotation.arg1
                    arg2 = re_annotation.arg2

                    arg1_tag = id2annotation[arg1].tag
                    arg2_tag = id2annotation[arg2].tag
                    start_token_arg1 = id2annotation[arg1].start_token_pos - total_token_dumped
                    end_token_arg1 = id2annotation[arg1].end_token_pos - total_token_dumped
                    start_token_arg2 = id2annotation[arg2].start_token_pos - total_token_dumped
                    end_token_arg2 = id2annotation[arg2].end_token_pos - total_token_dumped

                    token_in_dump = len(dump["input_ids"])
                    if (
                        start_token_arg1 >= 0
                        and start_token_arg2 >= 0
                        and end_token_arg1 >= 0
                        and end_token_arg2 >= 0
                        and start_token_arg1 < token_in_dump
                        and start_token_arg2 < token_in_dump
                        and end_token_arg1 < token_in_dump
                        and end_token_arg2 < token_in_dump
                    ):
                        dump_relations["relations"].append(
                            {
                                "arg1_tag": arg1_tag,
                                "arg2_tag": arg2_tag,
                                "arg1_pos": [start_token_arg1, end_token_arg1],
                                "arg2_pos": [start_token_arg2, end_token_arg2],
                                "re_tag": re_annotation.tag,
                            }
                        )
                        relations_count += 1
                relations.append(copy.deepcopy(dump_relations))
                dump["id"] = text_id
                tokenized_texts.append(copy.deepcopy(dump))
                total_token_dumped += len(dump["input_ids"])
                text_id += 1

                for key in ["input_ids", "labels", "text_labels"]:
                    dump[key].clear()
        skipped_relations += len(re_annotations) - relations_count

    label2id = get_mapping_to_id(args.label2id, labels_set)
    retag2id = get_mapping_to_id(args.retag2id, retags_set)

    for i in range(len(tokenized_texts)):
        tokenized_texts[i]["labels"] = [label2id[label] for label in tokenized_texts[i]["text_labels"]]
    for i in range(len(relations)):
        for j in range(len(relations[i]["relations"])):
            relations[i]["relations"][j]["tag"] = retag2id[relations[i]["relations"][j]["re_tag"]]

    save_jsonl(tokenized_texts, os.path.join(args.dir, "labeled_texts.jsonl"))
    save_jsonl(relations, os.path.join(args.dir, "relations.jsonl"))
    save_json(label2id, os.path.join(args.dir, "label2id.json"))
    save_json(retag2id, os.path.join(args.dir, "retag2id.json"))
    print(skipped_relations)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(_args)
