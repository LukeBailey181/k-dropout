import random
import argparse
from tqdm import tqdm
import sys

import torch
import torch.nn as nn
import numpy as np
import wandb

from wandb_helpers import write_git_snapshot

from k_dropout.networks import make_net
from k_dropout.training_helpers import test_net
from k_dropout.experiment_helpers import get_dataset
from k_dropout.modules import SequentialKDropout


def use_manual_seed(model: nn.Sequential, seed: int):
    for layer in model:
        if isinstance(layer, SequentialKDropout):
            layer.use_manual_seed = True
            layer.manual_seed = seed


def remove_manual_seed(model: nn.Sequential):
    for layer in model:
        if isinstance(layer, SequentialKDropout):
            layer.use_manual_seed = False


if __name__ == "__main__":
    """
    Train a sequential dropout network on cifar10 and track the performance of
    the dropout mask and random subnets throughout training.
    """

    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n_random_subnets", type=int, default=10)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=229)
    # training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    # model
    parser.add_argument("--input_size", type=int, default=3072)
    parser.add_argument("--hidden_size", type=int, default=2000)
    parser.add_argument("--n_hidden", type=int, default=2)
    parser.add_argument("--output_size", type=int, default=10)
    # dropout layer (always sequential for this experiment)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--input_p", type=float, default=0.0)
    parser.add_argument(
        "--m", type=int, default=1
    )  # m != 1 is more difficult to interpret
    # dataset
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # wandb integration
    # log the git diff and untracked files as an artifact
    snapshot_name, snapshot_path = write_git_snapshot()

    config = vars(args)
    config["git_snapshot"] = snapshot_name

    if args.run_name is None:
        run_name = f"sequential_subnet_k={args.k}_epochs={args.epochs}"
    else:
        run_name = args.run_name
    run = wandb.init(project="k-dropout", config=config, name=run_name)

    snapshot_artifact = wandb.Artifact(snapshot_name, type="git_snapshot")
    snapshot_artifact.add_file(snapshot_path)
    wandb.log_artifact(snapshot_artifact)

    # create model
    dropout_layer = SequentialKDropout
    layer_kwargs = {
        "p": args.p,
        "k": args.k,
        "m": args.m,
    }

    input_dropout_kwargs = dict(layer_kwargs)
    input_dropout_kwargs["p"] = args.input_p

    model = make_net(
        input_size=args.input_size,
        output_size=args.output_size,
        hidden_size=args.hidden_size,
        n_hidden=args.n_hidden,
        dropout_layer=dropout_layer,
        dropout_kwargs=layer_kwargs,
        input_dropout_kwargs=input_dropout_kwargs,
    )
    print(model)

    # get dataset
    train_set, test_set = get_dataset(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        preprocess_dataset=True,
        device=args.device,
    )
    print(f"Loaded {args.dataset_name} dataset")

    # train
    print(f"Training for {args.epochs} epochs...")

    # setup manual seeds
    n_batches = len(train_set)
    assert args.k % n_batches == 0
    total_subnets = int(args.epochs * n_batches / args.k)
    epochs_per_subnet = int(args.epochs / total_subnets)

    mask_subnet_seeds = np.random.randint(2**32, size=total_subnets).tolist()
    random_subnet_seeds = np.random.randint(
        2**32, size=args.n_random_subnets
    ).tolist()

    # custom training loop with manual seeding
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.to(args.device)

    example_ct = 0
    for epoch in tqdm(range(args.epochs)):
        mask_seed_ix = epoch // epochs_per_subnet
        use_manual_seed(model, mask_subnet_seeds[mask_seed_ix])

        model.train()
        epoch_loss = 0
        for X, y in train_set:
            example_ct += X.shape[0]
            X = X.to(args.device)
            y = y.to(args.device)

            model.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            wandb.log({"train_batch_loss": loss.item()}, step=example_ct)
            epoch_loss += loss.item()
        wandb.log({"train_epoch_loss": epoch_loss}, step=example_ct)

        # evaluate...
        # for each mask subnet
        for ix, seed in enumerate(mask_subnet_seeds):
            use_manual_seed(model, seed)
            test_loss, acc = test_net(model, test_set, device=args.device)
            wandb.log(
                {f"test_loss_mask_{ix}": test_loss, f"test_acc_mask_{ix}": acc},
                step=example_ct,
            )

        # for each random subnet
        for ix, seed in enumerate(random_subnet_seeds):
            use_manual_seed(model, seed)
            test_loss, acc = test_net(model, test_set, device=args.device)
            wandb.log(
                {f"test_loss_random_{ix}": test_loss, f"test_acc_random_{ix}": acc},
                step=example_ct,
            )

        # for the entire model
        remove_manual_seed(model)
        test_loss, acc = test_net(model, test_set, device=args.device)
        wandb.log({"test_loss_full": test_loss, "test_acc_full": acc}, step=example_ct)
