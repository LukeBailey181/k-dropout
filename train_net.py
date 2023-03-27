import random

import torch
import numpy as np
import wandb

from wandb_helpers import write_git_snapshot

from k_dropout.networks import make_net
from k_dropout.training_helpers import train_net
from k_dropout.experiment_helpers import (
    get_default_parser,
    get_dataset,
    get_dropout_layer,
)


if __name__ == "__main__":
    """
    Train a network according to the parameters specified at the command line
    and log the results to weights and biases.
    """

    parser = get_default_parser()
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
        m=args.m,
        cache_masks=args.cache_masks,
        hidden_size=args.hidden_size,
    )
    model = make_net(
        input_size=args.input_size,
        output_size=args.output_size,
        hidden_size=args.hidden_size,
        n_hidden=args.n_hidden,
        dropout_layer=dropout_layer,
        dropout_kargs=layer_kwargs,
    )
    print(f"Created {args.dropout_layer} model with {args.n_hidden} hidden layers")
    print(model)

    # get dataset
    train_set, test_set = get_dataset(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        preprocess_dataset=args.preprocess_dataset,
        device=args.device,
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
