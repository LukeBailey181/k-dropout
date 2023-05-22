import torch
from torch import nn
from tqdm import tqdm
import wandb


if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# TODO: handle other metrics (from sklearn?)
def test_net(net, dataset, device=DEVICE, eval_net=True):
    """
    Evaulates net on dataset

    Arguments:
        net: network to test
        dataset: dataset to test net on
        device: device to move net and data to, e.g. 'cuda'
        eval_net: true if you want 'net.eval()' to be run. If not, the 
            net mode (train or eval) is unchaged. Used if you want to 
            keep a net with dropout in in training mode to evaluate a 
            subnet performance.
    """

    criterion = nn.CrossEntropyLoss()

    net.to(device)
    if eval_net:
        net.eval()

    total_loss = total_correct = total_examples = 0
    with torch.no_grad():
        for X, y in dataset:
            output = nn.functional.softmax(net(X), dim=-1)
            loss = criterion(output, y)

            total_loss += loss.item()
            total_correct += (output.argmax(dim=-1) == y).sum().item()
            total_examples += y.shape[0]

    return total_loss, total_correct / total_examples


def train_net(
    net,
    train_set,
    epochs: int,
    lr: float = 0.005,
    device=DEVICE,
    test_set=None,
    eval_every: int = 1,
    use_wandb: bool = False,
    return_results: bool = True,
):
    """
    Trains net using provided train_set.

    Arguments:
        net: network to train
        train_set: data_set to train on
        epochs: number of epochs to train for
        test_set: data_set to test on
        eval_every: test on the test_set every eval_every epochs
        lr: learning rate
        preproc: whether or not to move the data_sets onto the specified device
            before training and testing
        device: device to train and test on
        use_wandb: whether or not to log to weights and biases
        return_results: whether or not to return the results of training

    Returns (if return_results == True) dict containing:
        train_epoch_losses: list of losses for each epoch
        train_batch_losses: list of losses for each batch
        test_losses: dict mapping epoch to test loss
        test_accs: dict mapping epoch to test accuracy
    """

    # TODO: optionally use wandb.watch() to log gradients and parameters

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    net.to(device)

    batch_losses = []
    epoch_losses = []
    test_losses = {}
    test_accs = {}
    example_ct = 0  # track number of examples seen for logging
    for epoch in tqdm(range(epochs)):
        net.train()
        epoch_loss = 0
        for X, y in train_set:
            X = X.to(device)
            y = y.to(device)

            net.zero_grad()
            output = net(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            example_ct += X.shape[0]
            if return_results:
                batch_losses.append(loss.item())
            if use_wandb:
                wandb.log({"train_batch_loss": loss.item()}, step=example_ct)
            epoch_loss += loss.item()

        if return_results:
            epoch_losses.append(epoch_loss)
        if use_wandb:
            wandb.log({"train_epoch_loss": epoch_loss}, step=example_ct)

        # evaluate
        if test_set is not None and epoch % eval_every == 0:
            if return_results:
                test_loss, acc = test_net(net, test_set)
                test_losses[epoch] = test_loss
                test_accs[epoch] = acc
            if use_wandb:
                wandb.log({"test_loss": test_loss, "test_acc": acc}, step=example_ct)

    # final test of model
    if test_set is not None and (epochs - 1) not in test_losses:
        if return_results:
            test_loss, acc = test_net(net, test_set)
            test_losses[epochs - 1] = test_loss
            test_accs[epochs - 1] = acc
        if use_wandb:
            wandb.log({"test_loss": test_loss, "test_acc": acc}, step=example_ct)

    if return_results:
        return {
            "train_epoch_losses": epoch_losses,
            "train_batch_losses": batch_losses,
            "test_losses": test_losses,
            "test_accs": test_accs,
        }
