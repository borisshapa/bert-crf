import os.path
from argparse import ArgumentParser, Namespace

import numpy as np
import torch.nn
import torch.utils.data as td
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertForMaskedLM

from datasets import MaskedLanguageModelingDataset, Subset


def configure_arg_parser():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--dataset",
        type=str,
        default="resources/data/train/masked_texts.json",
        help="Path to JSON where the source data is stored",
    )
    arg_parser.add_argument(
        "--model-name",
        type=str,
        default="sberbank-ai/ruBert-base",
        help="The name of the model. "
             "This can be a model from the Hugging Face pub or a local path.",
    )
    arg_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device which is used for training"
    )
    arg_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Amount of epochs to train"
    )
    arg_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate of AdamW optimizer. (default: 1e-5)"
    )
    arg_parser.add_argument(
        "--valid-size",
        type=float,
        default=0.15,
        help="Validation dataset percentage size. (default: 0.15)"
    )
    arg_parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size. (default: 128)"
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

    dataset = MaskedLanguageModelingDataset(jsonl_file=args.dataset, device=args.device)

    train_indices, valid_indices = train_test_split(np.arange(len(dataset)), test_size=args.test_size)

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)

    train_loader = td.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=dataset.collate_function)
    valid_loader = td.DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=dataset.collate_function)

    model = BertForMaskedLM.from_pretrained(args.model_name).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_losses = []
    valid_losses = []

    for e in range(args.epochs):
        train_loss, valid_loss = epoch(model, optimizer, train_loader, valid_loader)

        if np.min(valid_losses, initial=np.Inf) < valid_loss:
            print(f"Overfitting! Training loop is finished at {e + 1} epoch")
            break

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

    torch.save(model, "xxx.pth")
    model.save_pretrained("ruREBus/ruBert-base")


def epoch(
        model: BertForMaskedLM,
        optimizer: torch.optim.Optimizer,
        train_loader: td.DataLoader,
        valid_loader: td.DataLoader
):
    train_loss, valid_loss = 0, 0

    model.train()
    for batch in tqdm(train_loader, desc="training"):
        output = model.forward(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().cpu().item()

    model.eval()
    for batch in tqdm(valid_loader, desc="validation"):
        with torch.no_grad():
            output = model.forward(**batch)
            loss = output.loss
            valid_loss += loss.detach().cpu().item()

    return train_loss / len(train_loader), valid_loss / len(valid_loader)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(_args)
