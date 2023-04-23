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
    if not args.local_only:
        # log the git diff and untracked files as an artifact
        
        #snapshot_name, snapshot_path = write_git_snapshot()

        config = vars(args)
        #config["git_snapshot"] = snapshot_name
        # TODO: specify run name in a more precise way
        #       e.g. with args.run_name_prefix and args.run_name_items which could
        #       include the params (p, k, etc..) to put in the run name
        if args.run_name is None:
            run_name = f"{args.dataset_name}_{args.dropout_layer}"
            if args.dropout_layer == "sequential":
                run_name += f"_k={args.k}_m={args.m}"
            elif args.dropout_layer == "pool":
                run_name += f"_size={args.pool_size}_m={args.m}"
            run_name += f"_p={args.p}_lr={args.lr}_epochs={args.epochs}"
            if args.input_p:
                run_name += f"_input_p={args.input_p}"
        else:
            run_name = args.run_name
        run = wandb.init(project="k-dropout", config=config, name=run_name)

        # TODO fix artifact logging
        #snapshot_artifact = wandb.Artifact(snapshot_name, type="git_snapshot")
        #snapshot_artifact.add_file(snapshot_path)
        #wandb.log_artifact(snapshot_artifact)

    # create model
    dropout_layer, layer_kwargs = get_dropout_layer(
        dropout_layer=args.dropout_layer,
        p=args.p,
        k=args.k,
        pool_size=args.pool_size,
        sync_over_model=args.sync_over_model,
        m=args.m,
        cache_masks=True,
        hidden_size=args.hidden_size,
    )

    if args.input_p is not None:
        input_dropout_kwargs = dict(layer_kwargs)
        input_dropout_kwargs["p"] = args.input_p
        if args.dropout_layer == "pool":
            input_dropout_kwargs["input_dim"] = args.input_size
    else:
        input_dropout_kwargs = None

    model = make_net(
        input_size=args.input_size,
        output_size=args.output_size,
        hidden_size=args.hidden_size,
        n_hidden=args.n_hidden,
        dropout_layer=dropout_layer,
        dropout_kwargs=layer_kwargs,
        input_dropout_kwargs=input_dropout_kwargs,
    )
    print(f"Created {args.dropout_layer} model with {args.n_hidden} hidden layers")
    print(model)

    # get dataset
    train_set, test_set = get_dataset(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        preprocess_dataset=True,  # TODO: handle this without store_true
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
        use_wandb=not args.local_only,
    )

    if args.model_save_path is not None:
        # Save trained model 
        torch.save(model, args.model_save_path)

