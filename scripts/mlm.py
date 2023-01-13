import os.path
from argparse import ArgumentParser, Namespace

import numpy as np
import torch.nn
import torch.optim.lr_scheduler
import torch.utils.data as td
import transformers
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets import JsonlDataset, Subset
from re_utils.train import dict_to_device


def configure_arg_parser():
    arg_parser = ArgumentParser(
        description="Applies Masked Language Modeling (MLM) for BERT model using masked tokenized text in "
                    "unsupervised manner. Use MLM for finetunning your model and shift model weight onto specific domain."
    )
    arg_parser.add_argument(
        "--dataset",
        type=str,
        default="resources/data/train/masked_texts.json",
        help="Path to .jsonl file where the source data is stored",
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
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate of AdamW optimizer. (default: 1e-4)"
    )
    arg_parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.975,
        help="StepLR scheduler gamma coefficient. (default: 0.975)"
    )
    arg_parser.add_argument(
        "--lr-step-size",
        type=int,
        default=500,
        help="StepLR scheduler step size, after given amount of steps will multiple learning rate by given gamma. "
             "(default: 500)"
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
        default=32,
        help="Batch size. (default: 32)"
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

    dataset = JsonlDataset(dataset_path=args.dataset)

    train_indices, valid_indices = train_test_split(np.arange(len(dataset)), test_size=args.valid_size)

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)

    train_loader = td.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=dataset.collate_function)
    valid_loader = td.DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=dataset.collate_function)

    bert = transformers.BertForMaskedLM.from_pretrained(args.model_name).to(args.device)
    optimizer = torch.optim.AdamW(bert.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    train_losses = []
    valid_losses = []

    for e in range(args.epochs):
        train_loss, valid_loss = epoch(bert, optimizer, scheduler, train_loader, valid_loader, args.device)

        print(f"==> epoch: {e} finished | train loss: {train_loss:.5f} | valid loss: {valid_loss:.5f}")

        if valid_loss < np.min(valid_losses, initial=np.Inf):
            print("==> checkpoint updated <==")
            bert.save_pretrained(os.path.join(args.save_to, "ruREBus-bert"))
        else:
            print(f"Overfitting! Training loop is finished at {e + 1} epoch")
            break

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print()

    with open("logs.csv", "w") as f:
        logs = ["epoch,train,valid"]
        logs += [f"{i},{tl},{vl}" for i, (tl, vl) in enumerate(zip(train_losses, valid_losses))]
        logs = "\n".join(logs)
        f.write(logs)


def epoch(
        model: transformers.BertForMaskedLM,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.StepLR,
        train_loader: td.DataLoader,
        valid_loader: td.DataLoader,
        device: str
):
    train_loss, valid_loss = 0, 0

    model.train()
    bar = tqdm(train_loader, desc="training")
    for batch in bar:
        batch = dict_to_device(batch, device=device)
        optimizer.zero_grad()
        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.detach().cpu().item()
        bar.set_postfix_str(f"loss: {loss.detach().cpu().item():.4f}")

    model.eval()
    bar = tqdm(valid_loader, desc="validation")
    for batch in bar:
        with torch.no_grad():
            batch = dict_to_device(batch, device)
            output = model(**batch)
            loss = output.loss
            valid_loss += loss.detach().cpu().item()
            bar.set_postfix_str(f"loss: {loss.detach().cpu().item():.4f}")

    return train_loss / len(train_loader), valid_loss / len(valid_loader)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(_args)
