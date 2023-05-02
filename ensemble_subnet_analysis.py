import torch
import random
import numpy as np
from torch.nn import Sequential
import wandb
from statistics import mean, mode
import argparse

from k_dropout.experiment_helpers import DATASETS, get_dataset
from k_dropout.networks import PoolDropoutLensNet
from k_dropout.training_helpers import test_net, train_net

BATCH_SIZE = 512

def evaluate_ensemble_of_subnets(
    path_to_load_model, 
    subnet_idx, 
    test_set, 
    train_set, 
    epochs, 
    lr, 
    device, 
    seed,
    path_to_save_model=None,
):
    # seed experiment
    # TODO just move to __main__
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Init lens net using the loaded model
    net = torch.load(path_to_load_model)
    lens_net = PoolDropoutLensNet(init_net=net)
    lens_net.reset_weights()

    lens_net.freeze_mask(subnet_idx)

    train_net(
        net=net,
        train_set=train_set,
        test_set=test_set,
        epochs=epochs,
        lr=lr,
        device=device,
        use_wandb=True, # TODO handle local runs
    )

    if path_to_save_model is not None:
        torch.save(lens_net.net, path_to_save_model + f"_{subnet_idx}")

def evaluate_ensemble(
    path_to_models_dir, 
    model_prefix,
    num_models,
    test_set
):
    
    models = []
    for subnet_idx in range(num_models):
        # Get string 
        path = path_to_models_dir + model_prefix + f"_{subnet_idx}"
        net = torch.load(path)
        lens_net = PoolDropoutLensNet(init_net=net)
        lens_net.freeze_mask(subnet_idx)

        models.append(torch.load(lens_net))

    # Evaluate ensemble
    with torch.no_grad():

        num_correct = 0
        total_examples = 0

        for X, y in test_set:
            # Make sure batch size is 1
            assert(X.shape[0] == 1)
            preds = []
            for lens_net in models:
                # Apprend predictions to preds
                output = lens_net(X)
                preds.append(output.argmax(dim=1).item())
            
            pred = mode(preds)
            num_correct += pred == y.item() 
            total_examples += 1 

        print(num_correct/total_examples)

if __name__=="__main__":
    # TODO do some arg parsing here
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Experiment
    parser.add_argument(
        "--path_to_load_model",
        type=str,
        required=True,
        help="Path to the model that experiment will be run on",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=DATASETS,
        required=True,
        help="Name of the dataset to use for testing during experiment",
    )
    parser.add_argument(
        "--subnet_idx",
        type=int,
        required=True,
        help="Index of the subnet from model at --path_to_model to be trained and tested",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)


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
    parser.add_argument(
        "--path_to_save_model",
        type=str,
        help="Path to save the models to. Subnet idx will be appended. If not given then model is not saved",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randrange(2**32),
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    config = vars(args)

    run_name = f"{args.dataset_name}_ensemble_subnet_{args.subnet_idx}"
    # TODO add support for local run
    run = wandb.init(project="k-dropout", config=config, name=run_name)

    # get dataset
    train_set, test_set = get_dataset(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        num_workers=args.num_workers,
        preprocess_dataset=args.preprocess_dataset,
        device=args.device,
    )

    evaluate_ensemble_of_subnets(
        path_to_load_model=args.path_to_load_model, 
        subnet_idx=args.subnet_idx, 
        test_set=test_set, 
        train_set=train_set, 
        epochs=args.epochs, 
        lr=args.lr, 
        device=args.device, 
        seed=args.seed,
        path_to_save_model=args.path_to_save_model,
    )


    """
    train_set, test_set = get_dataset(
        dataset_name="cifar10",
        batch_size=1,
        test_batch_size=1,
        num_workers=4,
        preprocess_dataset=True,
        device="cuda",
    )
    evaluate_ensemble(
        path_to_models_dir="./models/ensemble/cifar10",
        model_prefix="cifar10_ensemble_subnet_subnet",
        num_models=50,
        test_set=test_set
    )
    """