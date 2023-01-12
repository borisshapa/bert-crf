import itertools
import json
import os.path
from argparse import Namespace, ArgumentParser

import torch
from tqdm import tqdm

from models.bert_crf import BertCrf
from re_utils.common import load_jsonl


def configure_arg_parser():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--bert-crf-path",
        type=str,
        default="weights/bert-crf.pt",
        help="The path to the file with bert crf weights",
    )
    arg_parser.add_argument(
        "--labeled-texts",
        type=str,
        default="resources/data/train/labeled_texts.jsonl",
        help="the path to the file with preprocessed texts",
    )
    arg_parser.add_argument(
        "--relations",
        type=str,
        default="resources/data/train/relations.jsonl",
        help="the path to the file with relations between entities",
    )
    arg_parser.add_argument(
        "--num-labels", type=int, default=17, help="number of possible tags for tokens"
    )
    arg_parser.add_argument(
        "--bert-name",
        type=str,
        default="sberbank-ai/ruBert-base",
        help="local path or hf hub BERT name that underlies the model",
    )
    arg_parser.add_argument(
        "--bert-dropout", type=float, default=0.2, help="dropout probability"
    )
    arg_parser.add_argument(
        "--use-crf",
        type=bool,
        default=True,
        help="whether use conditional random field or not",
    )
    arg_parser.add_argument(
        "--label2id",
        type=str,
        default="resources/data/train/label2id.json",
        help="Json file with mapping from entity label to its id",
    )
    arg_parser.add_argument(
        "--retag2id",
        type=str,
        default="resources/data/train/retag2id.json",
        help="Json file with mapping from relation tag to its id",
    )
    return arg_parser


def main(args: Namespace):
    dir = os.path.dirname(args.labeled_texts)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BertCrf(
        num_labels=args.num_labels,
        bert_name=args.bert_name,
        dropout=args.bert_dropout,
        use_crf=args.use_crf,
    )
    model.load_from(args.bert_crf_path)
    model.eval()
    model = model.to(device)

    labeled_texts = load_jsonl(args.labeled_texts)
    relations = load_jsonl(args.relations)

    with open(args.label2id, "r") as label2id_file:
        label2id = json.load(label2id_file)
    id2label = {id: label for label, id in label2id.items()}

    with open(args.retag2id, "r") as retag2id_file:
        retag2id = json.load(retag2id_file)
    no_relation_tag = len(retag2id)

    tag_counter = {tag: 0 for tag in retag2id.values()}
    tag_counter[no_relation_tag] = 0

    with open(
        os.path.join(dir, "relation_training_data.jsonl"), "w"
    ) as relation_training_data_file:
        for labeled_text, text_relations in tqdm(list(zip(labeled_texts, relations))):
            assert labeled_text["id"] == text_relations["id"]
            input_ids = torch.tensor([labeled_text["input_ids"]], device=device)
            attention_mask = torch.ones(1, len(labeled_text["input_ids"]), device=device)

            _, batched_bert_embeddings = model.get_bert_features(
                input_ids, attention_mask
            )
            bert_embeddings = batched_bert_embeddings[0]
            full_seq_embedding = bert_embeddings.mean(dim=0).tolist()
            labels = model.decode(input_ids, attention_mask)[0]

            labels_pos = []
            ind = 0
            while ind < len(labels):
                if id2label[labels[ind]].startswith("B"):
                    tag = id2label[labels[ind]].split("-")[1]
                    start_pos = ind
                    ind += 1
                    while ind < len(labels) and id2label[labels[ind]].startswith("I"):
                        ind += 1
                    end_pos = ind
                    labels_pos.append({"tag": tag, "pos": [start_pos, end_pos]})
                else:
                    ind += 1

            all_pairs = itertools.permutations(labels_pos, 2)
            for first_arg, second_arg in all_pairs:
                first_arg_embedding = bert_embeddings[
                    first_arg["pos"][0] : first_arg["pos"][1]
                ].mean(dim=0).tolist()
                second_arg_embedding = bert_embeddings[
                    second_arg["pos"][0] : second_arg["pos"][1]
                ].mean(dim=0).tolist()

                relation_tag = no_relation_tag
                for relation in text_relations["relations"]:
                    if (
                        relation["arg1_tag"] == first_arg["tag"]
                        and relation["arg2_tag"] == second_arg["tag"]
                        and relation["arg1_pos"] == first_arg["pos"]
                        and relation["arg2_pos"] == second_arg["pos"]
                    ):
                        relation_tag = relation["tag"]
                        break

                tag_counter[relation_tag] += 1
                json.dump(
                    {
                        "seq_emb": full_seq_embedding,
                        "arg1_emb": first_arg_embedding,
                        "arg2_emb": second_arg_embedding,
                        "tag": relation_tag,
                    },
                    relation_training_data_file,
                )
                relation_training_data_file.write("\n")

    print(tag_counter)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(_args)
