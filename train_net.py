import argparse
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import wandb

from wandb_helpers import write_git_snapshot

from k_dropout.networks import make_net
from k_dropout.datasets import get_mnist, get_cifar10, process_dataset
from k_dropout.modules import SequentialKDropout, PoolKDropout
from k_dropout.training_helpers import train_net


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASETS = ("mnist", "cifar10")
DROPOUT_LAYERS = ("none", "standard", "sequential", "pool")


def get_dropout_layer(
    dropout_layer: str,
    p: float,
    k: Optional[int] = None,
    pool_size: Optional[int] = None,
    masks_per_batch: Optional[int] = None,
) -> Tuple[nn.Module, dict]:

    kwargs = {"p": p}
    if masks_per_batch is not None:
        kwargs["m"] = masks_per_batch

    if dropout_layer == "none":
        return None, None
    elif dropout_layer == "standard":
        return torch.nn.Dropout, kwargs
    elif dropout_layer == "sequential":
        if k is None:
            raise ValueError("Must specify k for sequential dropout")
        kwargs["k"] = k
        return SequentialKDropout, kwargs
    elif dropout_layer == "pool":
        if pool_size is None:
            raise ValueError("Must specify pool_size for pool dropout")
        # TODO change this to be pool size everywhere
        kwargs["n_masks"] = pool_size
        return PoolKDropout, kwargs
    else:
        raise ValueError(f"Unknown dropout layer {dropout_layer}")


def get_dataset(
    dataset_name: str,
    batch_size: int,
    test_batch_size: int,
    num_workers: int,
    preprocess: bool,
    device: str,
):
    if dataset_name == "mnist":
        train_loader, test_loader = get_mnist(batch_size, test_batch_size, num_workers)
    elif dataset_name == "cifar10":
        train_loader, test_loader = get_cifar10(
            batch_size, test_batch_size, num_workers
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if preprocess:
        return process_dataset(train_loader, device), process_dataset(
            test_loader, device
        )
    return train_loader, test_loader


if __name__ == "__main__":
    """
    Train a network according to the parameters specified at the command line
    and log the results to weights and biases.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # model
    parser.add_argument(
        "--dropout_layer",
        type=str,
        choices=DROPOUT_LAYERS,
        required=True,
        help="Type of dropout layer to use",
    )
    parser.add_argument(
        "--input_size", type=int, required=True, help="Dimension of the model input"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        required=True,
        help="Dimension of the model hidden layer(s)",
    )
    parser.add_argument(
        "--n_hidden", type=int, required=True, help="Number of hidden layers"
    )
    parser.add_argument(
        "--output_size", type=int, required=True, help="Dimension of the model output"
    )
    parser.add_argument("--p", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--k", type=int, help="Mask repeat parameter for k dropout")
    parser.add_argument(
        "--pool_size", type=int, help="Number of masks in the pool for pool k dropout"
    )
    parser.add_argument(
        "--masks_per_batch",
        type=int,
        help="Number of submasks to include in each batch for k dropout",
    )
    # dataset
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=DATASETS,
        required=True,
        help="Name of the dataset to use for training and evaluation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=128, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--preprocess_dataset",
        action="store_true",  # defaults to false
        help="Move the entire dataset to the device before training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers for the data loader",
    )
    # training
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=DEVICE)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    # experiment
    parser.add_argument(
        "--use_wandb",
        action="store_true",  # defaults to false
        help="Whether to log results to weights and biases",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of the run to use for weights and biases",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randrange(2**32),
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # wandb integration
    if args.use_wandb:
        # log the git diff and untracked files as an artifact
        snapshot_name, snapshot_path = write_git_snapshot()

        config = vars(args)
        config["git_snapshot"] = snapshot_name
        run = wandb.init(project="k-dropout", config=config, name=args.run_name)

        snapshot_artifact = wandb.Artifact(snapshot_name, type="git_snapshot")
        snapshot_artifact.add_file(snapshot_path)
        wandb.log_artifact(snapshot_artifact)

    # create model
    dropout_layer, layer_kwargs = get_dropout_layer(
        dropout_layer=args.dropout_layer,
        p=args.p,
        k=args.k,
        pool_size=args.pool_size,
        masks_per_batch=args.masks_per_batch,
    )
    model = make_net(
        input_dim=args.input_size,
        num_classes=args.output_size,
        hidden_units=args.hidden_size,
        hidden_layers=args.n_hidden,
        dropout_layer=dropout_layer,
        dropout_kargs=layer_kwargs,
    )
    print(f"Created {args.dropout_layer} model with {args.n_hidden} hidden layers")
    print(model)

    # get dataset
    train_set, test_set = get_dataset(
        args.dataset_name,
        args.batch_size,
        args.test_batch_size,
        args.num_workers,
        args.preprocess_dataset,
        args.device,
    )
    print(f"Loaded {args.dataset_name} dataset")

    # train
    print(f"Training for {args.epochs} epochs...")
    train_net(
        net=model,
        train_set=train_set,
        test_set=test_set,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        use_wandb=args.use_wandb,
    )
