import argparse
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn

from k_dropout.datasets import get_mnist, get_cifar10, process_dataset
from k_dropout.modules import SequentialKDropout, PoolKDropout


DATASETS = ("mnist", "cifar10")
DROPOUT_LAYERS = ("none", "standard", "sequential", "pool")


def get_default_parser():
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
    # k dropout common
    parser.add_argument("--p", type=float, default=0.5, help="Dropout probability")
    parser.add_argument(
        "--m",
        type=int,
        help="Number of submasks to include in each batch for k dropout",
    )
    # sequential k dropout
    parser.add_argument("--k", type=int, help="Mask repeat parameter for k dropout")
    # pool k dropout
    parser.add_argument(
        "--pool_size", type=int, help="Number of masks in the pool for pool k dropout"
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
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    # experiment
    parser.add_argument(
        "--local_only",
        action="store_true",  # defaults to false
        help="If true, don't use weights and biases",
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
    # TODO: find a better way to do restarts in sweeps
    parser.add_argument(
        "--restarts", type=str, help="Placeholder for running multiple sweep restarts"
    )
    return parser


def get_dropout_layer(
    dropout_layer: str,
    p: float,
    k: Optional[int] = None,
    pool_size: Optional[int] = None,
    m: Optional[int] = None,
    cache_masks: bool = True,
    hidden_size: Optional[int] = None,  # input size for pool dropout with mask caching
) -> Tuple[nn.Module, dict]:
    kwargs = {"p": p}
    if m is not None:
        kwargs["m"] = m

    if dropout_layer == "none":
        return None, None
    elif dropout_layer == "standard":
        return nn.Dropout, kwargs
    elif dropout_layer == "sequential":
        if k is None:
            raise ValueError("Must specify k for sequential dropout")
        kwargs["k"] = k
        return SequentialKDropout, kwargs
    elif dropout_layer == "pool":
        if pool_size is None:
            raise ValueError("Must specify pool_size for pool dropout")
        kwargs["pool_size"] = pool_size
        kwargs["cache_masks"] = cache_masks
        if cache_masks:
            kwargs["input_dim"] = hidden_size
        return PoolKDropout, kwargs
    else:
        raise ValueError(f"Unknown dropout layer {dropout_layer}")


def get_dataset(
    dataset_name: str,
    batch_size: int,
    test_batch_size: int,
    num_workers: int,
    preprocess_dataset: bool,
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

    if preprocess_dataset:
        return process_dataset(train_loader, device), process_dataset(
            test_loader, device
        )
    return train_loader, test_loader
