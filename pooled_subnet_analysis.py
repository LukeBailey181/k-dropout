import torch
import random
import numpy as np
from torch.nn import Sequential
import wandb
from statistics import mean
import argparse

from k_dropout.experiment_helpers import DATASETS, get_dataset
from k_dropout.networks import PoolDropoutLensNet
from k_dropout.training_helpers import test_net, train_net

"""
except ModuleNotFoundError:
    from experiment_helpers import DATASETS, get_dataset
    from networks import PoolDropoutLensNet
    from training_helpers import test_net
"""

EXPERIMENT_TYPES = ("pool_subnets", "standard_dropout_subnets")
BATCH_SIZE = 512

# TODO: implement plot_train_loss (using wandb and maybe without it too)
def plot_train_loss():
    pass


def evaluate_subnets_in_pooled_dropout_net(
    path_to_model, num_subnets, test_set, run_name, wandb_config
):

    # TODO add support for local run
    run = wandb.init(project="k-dropout", config=config, name=run_name)

    # Init lens net using the loaded model
    net = torch.load(path_to_model)
    lens_net = PoolDropoutLensNet(init_net=net)

    # Get pooled subnet accs
    pooled_subnet_accs = []
    for subnet_idx in range(num_subnets):
        lens_net.freeze_mask(subnet_idx)
        _, test_acc = test_net(lens_net, test_set)
        pooled_subnet_accs.append(test_acc)

    # Do wandb logging for pooled subnets
    subnet_idxs = list(range(num_subnets))
    wandb_plot(
        "pooled_subnet_accs",
        "Pooled Subnet Test Accuracy",
        "Subnet Index",
        "Test Accuracy",
        subnet_idxs,
        pooled_subnet_accs,
        "scatter",
    )

    # Get random subnet accs
    lens_net.activate_random_masking()
    print("Switched net to random droput:")
    print(lens_net)

    random_subnet_accs = []
    for _ in range(num_subnets):
        # Get accuracy when using dropout net by setting eval_net=False
        _, test_acc = test_net(lens_net, test_set, eval_net=False)
        random_subnet_accs.append(test_acc)

    # Do wandb logging for random subnets
    wandb_plot(
        "random_subnet_accs",
        "Random Subnet Test Accuracy",
        "Subnet Index",
        "Test Accuracy",
        subnet_idxs,
        random_subnet_accs,
        "scatter",
    )

    # Log average pool subnet, average random, full network
    lens_net.deactivate_random_masking()
    lens_net.unfreeze_mask()
    print("Switched net back to non random dropout:")
    print(lens_net)

    mean_pool_subnet = mean(pooled_subnet_accs)
    mean_random_subnet = mean(random_subnet_accs)
    _, full_test_acc = test_net(lens_net, test_set, eval_net=True)

    wandb_plot(
        plot_id="pool_subnets_summary",
        plot_title="Pool Subnet Summary",
        x_label="Net Type",
        y_label="Mean Test Accuracy",
        x_vals=["Pool Subnet", "Random Dropout Subnet", "Pool Net"],
        y_vals=[mean_pool_subnet, mean_random_subnet, full_test_acc],
        plot_type="bar",
    )


def evaluate_subnets_in_dropout_net(
    path_to_model, num_subnets, test_set, run_name, wandb_config
):

    # TODO add support for local run
    run = wandb.init(project="k-dropout", config=config, name=run_name)

    # Init lens net using the loaded model
    net = torch.load(path_to_model)
    net.train()

    dropout_subnet_accs = []
    for _ in range(num_subnets):
        # Get accuracy when using dropout net by setting eval_net=False
        _, test_acc = test_net(net, test_set, eval_net=False)
        dropout_subnet_accs.append(test_acc)

    subnet_idxs = list(range(num_subnets))
    net.eval()
    _, full_test_acc = test_net(net, test_set, eval_net=True)

    # Do wandb logging
    wandb_plot(
        "dropout_subnet_accs",
        "Dropout Subnet Test Accuracy",
        "Subnet Index",
        "Test Accuracy",
        subnet_idxs,
        dropout_subnet_accs,
        "scatter",
    )
    wandb_plot(
        plot_id="pool_subnets_summary",
        plot_title="Pool Subnet Summary",
        x_label="Net Type",
        y_label="Mean Test Accuracy",
        x_vals=["Dropout Net Subnet", "Dropout Net"],
        y_vals=[mean(dropout_subnet_accs), full_test_acc],
        plot_type="bar",
    )


def evaluate_ensemble_of_subnets(
    path_to_model, num_subnets, test_set, train_set, epochs, lr, seed, device
):
    # seed experiment
    # TODO just move to __main__
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Init lens net using the loaded model
    net = torch.load(path_to_model)
    lens_net = PoolDropoutLensNet(init_net=net)

    independent_subnet_accs = []
    for subnet_idx in range(num_subnets):
        lens_net.reset_weights()
        lens_net.freeze_mask(subnet_idx)

        train_net(
            net=net,
            train_set=train_set,
            test_set=test_set,
            epochs=epochs,
            lr=lr,
            device=device,
            use_wandb=True,  # TODO handle local runs
        )


def wandb_plot(plot_id, plot_title, x_label, y_label, x_vals, y_vals, plot_type):
    """Helper function for custom wandb plots

    Arguments:
        plot_id: string id of plot for wandb
        plot_title: string plot title
        x_label: x axis label
        y_label: y axis label
        x_vals: list of x values
        y_vals: list of y values
        plot_type: one of {line, scatter, bar} indicating style of plot.
    """

    if plot_type == "line":
        plot_func = wandb.plot.line
    elif plot_type == "scatter":
        plot_func = wandb.plot.scatter
    elif plot_type == "bar":
        plot_func = wandb.plot.bar
    else:
        raise RuntimeError(f"plot_style of {plot_type} not in ('line', 'scatter')")

    data = [[x, y] for (x, y) in zip(x_vals, y_vals)]
    table = wandb.Table(data=data, columns=[x_label, y_label])
    wandb.log({plot_id: plot_func(table, x_label, y_label, title=plot_title)})


if __name__ == "__main__":
    # TODO do some arg parsing here
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Experiment
    parser.add_argument(
        "--experiment_type",
        type=str,
        choices=EXPERIMENT_TYPES,
        required=True,
        help="Which experimnet to run",
    )
    parser.add_argument(
        "--path_to_model",
        type=str,
        required=True,
        help="Path to the model that experiment will be run on",
    )
    parser.add_argument(
        "--num_subnets",
        type=int,
        required=True,
        help="Number of subnets to evaluate during experiment",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=DATASETS,
        required=True,
        help="Name of the dataset to use for testing during experiment",
    )
    # Experiment execution
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers for the data loader",
    )
    parser.add_argument(
        "--preprocess_dataset",
        action="store_true",  # defaults to false
        help="Move the entire dataset to the device before training",
    )
    # wandb
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the run to use for weights and biases",
    )

    args = parser.parse_args()
    config = vars(args)

    # get dataset
    _, test_set = get_dataset(
        dataset_name=args.dataset_name,
        batch_size=BATCH_SIZE,
        test_batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
        preprocess_dataset=args.preprocess_dataset,
        device=args.device,
    )

    if args.experiment_type == "pool_subnets":
        evaluate_subnets_in_pooled_dropout_net(
            path_to_model=args.path_to_model,
            num_subnets=args.num_subnets,
            test_set=test_set,
            run_name=args.run_name,
            wandb_config=config,
        )
    elif args.experiment_type == "standard_dropout_subnets":
        evaluate_subnets_in_dropout_net(
            path_to_model=args.path_to_model,
            num_subnets=args.num_subnets,
            test_set=test_set,
            run_name=args.run_name,
            wandb_config=config,
        )
