import random
import argparse
from tqdm import tqdm
import sys
import math

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


def get_divisors(n):
    divisors = []
    for i in range(1, int(math.sqrt(n))):
        if(n % i == 0):
            divisors.append(i)
            divisors.append(n // i)

    return list(sorted(divisors))


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
    parser.add_argument("--skip_mask_performance", action='store_true')
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
    parser.add_argument("--run_suffix", type=str, default=None)

    # EXPERIMENT 13
    parser.add_argument("--disable_dropout_after_nth_subnet", type=int, default=None)

    # EXPERIMENT 14
    parser.add_argument("--log_param_magnitude_every_n", type=int, default=None)

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
        if(args.disable_dropout_after_nth_subnet is not None):
            run_name = f"disable_after_{args.disable_dropout_after_nth_subnet}_{run_name}"
    else:
        run_name = args.run_name

    if(args.run_suffix is not None):
        run_name = f"{run_name}_{args.run_suffix}"

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
    total_batches = args.epochs * n_batches
    total_subnets = (args.epochs * n_batches) // args.k)
    total_batches_to_run = (args.epochs * n_batches) * args.k

    mask_subnet_seeds = np.random.randint(2**32, size=total_subnets).tolist()
    random_subnet_seeds = np.random.randint(
        2**32, size=args.n_random_subnets
    ).tolist()

    # custom training loop with manual seeding
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.to(args.device)
    example_ct = 0

    def evaluate(skip_mask_performance=False):
        is_train = model.training

        # evaluate...
        # for each mask subnet
        if not skip_mask_performance:
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

        # test_net() flips on eval mode; undo if necessary
        if(is_train):
            model.train()

    evaluate(args.skip_mask_performance)  # evaluate once on the untrained model

    batch_ct = 0
    dropout_disabled = False
    for epoch in tqdm(range(args.epochs)):
        model.train()
        epoch_loss = 0
        for i, (X, y) in enumerate(train_set):
            if(batch_ct > total_batches_to_run):
                break

            mask_seed_ix = batch_ct // args.k
            use_manual_seed(model, mask_subnet_seeds[mask_seed_ix])

            if(
                (not dropout_disabled) and 
                args.disable_dropout_after_nth_subnet is not None and 
                mask_seed_ix + 1 > args.disable_dropout_after_nth_subnet
            ):
                print(f"Disabling dropout after {mask_seed_ix}-th subnet...")
                for layer in model.children():
                    if(isinstance(layer, SequentialKDropout)):
                        layer.disable()

                dropout_disabled = True

            batch_ct += 1
            example_ct += X.shape[0]
            X = X.to(args.device)
            y = y.to(args.device)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            wandb.log({"train_batch_loss": loss.item()}, step=example_ct)
            epoch_loss += loss.item()

            if(
                args.log_param_magnitude_every_n is not None and 
                i % args.log_param_magnitude_every_n == 0
            ):
                with torch.no_grad():
                    module_means = {}
                    total_param_count = 0
                    for n, p in model.named_parameters():
                        data = {}
                        data["mean"] = torch.mean(torch.abs(p))
                        data["param_count"] = torch.numel(p)
                        total_param_count += data["param_count"]
                        module_means[n] = data

                    cum_mean = 0
                    for n, d in module_means.items():
                        cum_mean += d["mean"] * (d["param_count"] / total_param_count)

                    wandb.log({"mean_param_magnitude": cum_mean}, step=example_ct)
 
        wandb.log({"train_epoch_loss": epoch_loss}, step=example_ct)

        evaluate(args.skip_mask_performance)  # evaluate after each epoch

#    import pickle
#    with open("model.pickle", "wb") as fp:
#        pickle.dump(model, fp, protocol=pickle.HIGHEST_PROTOCOL)
