import time
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

from networks import make_standard_net, make_skd_net


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def process_dataset(dataset, device=DEVICE):
    """
    Iterates through datset, applying transforms associated with dataloader and
    moving tensors onto device
    """
    preproc_data = []
    for batch in dataset:
        X, y = batch
        X = X.to(device)
        y = y.to(device)
        preproc_data.append([X, y])

    return preproc_data


def test_net(net, dataset, preproc=False, device=DEVICE):
    """
    Evaulates inputted net on inputted dataset
    """
    if preproc:
        dataset = process_dataset(dataset, device)

    criterion = nn.CrossEntropyLoss()

    net.to(device)
    net.eval()
    total_loss = total_correct = total_examples = 0
    with torch.no_grad():
        for data in dataset:
            X, y = data
            if not preproc:
                X = X.to(device)
                y = y.to(device)

            output = net(X)
            loss = criterion(output, y)
            total_loss += loss.item()
            total_correct += (output.argmax(dim=1) == y).sum().item()
            total_examples += len(y)

    return total_loss, total_correct / total_examples


def train_net(
    epochs,
    net,
    trainset,
    testset=None,
    eval_every=1,
    lr=0.005,
    plot=False,
    preproc=False,
    device=DEVICE,
):
    """
    Trains inputted net using provided trainset.
    """
    if preproc:
        trainset = process_dataset(trainset, device)
        if testset is not None:
            testset = process_dataset(testset, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)

    losses = []
    epoch_losses = []
    test_losses = {}
    test_accs = {}

    net.to(device)
    for epoch in tqdm(range(epochs)):

        net.train()
        epoch_loss = 0
        for data in trainset:
            X, y = data
            if not preproc:
                X = X.to(device)
                y = y.to(device)

            net.zero_grad()
            output = net(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            epoch_loss += loss.item()

        epoch_losses.append(epoch_loss)

        if testset is not None and epoch % eval_every == 0:
            # Evaluate model
            test_loss, acc = test_net(net, testset, preproc=False)
            test_losses[epoch] = test_loss
            test_accs[epoch] = acc

    if plot:
        plt.plot([i for i in range(len(losses[10:]))], losses[10:])
        plt.title("Training Loss")
        plt.xlabel("Batch")
        plt.show()

        plt.plot([i for i in range(len(epoch_losses))], epoch_losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.show()

    # Final test of model
    test_loss, acc = test_net(net, testset, preproc=False)
    test_losses[epochs - 1] = test_loss
    test_accs[epochs - 1] = acc

    return {
        "train_epoch_losses": epoch_losses,
        "train_batch_losses": losses,
        "test_losses": test_losses,
        "test_accs": test_accs,
    }


def get_mnist(train_batch_size=64, test_batch_size=1000, num_workers=0):
    """
    Download MNIST into ./datasets/ directory and return dataloaders containing
    MNIST. If ./datasets/ directory doesn't exist then it is made.

    Keyword arguments:
    train_batch_size -- size of batches in returned train set dataloader
    test_batch_size -- size of batches in returned test set dataloader
    """

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "./datasets/",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    torchvision.transforms.Lambda(lambda x: torch.flatten(x)),
                ]
            ),
        ),
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "./datasets/",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    torchvision.transforms.Lambda(lambda x: torch.flatten(x)),
                ]
            ),
        ),
        batch_size=test_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    return train_loader, test_loader


if __name__ == "__main__":

    # Train standard net
    train_loader, test_loader = get_mnist()
    standard_net = make_standard_net(num_classes=10, input_dim=784)
    train_net(10, standard_net, train_loader)

    # Train kdropout net
    train_loader, test_loader = get_mnist()
    dropout_net = make_skd_net(num_classes=10, input_dim=784, p=0.5, k=1)
    train_net(10, dropout_net, train_loader)

    _, standard_acc = test_net(standard_net, test_loader)
    _, dropout_acc = test_net(dropout_net, test_loader)
    print(f"Standard Net MNIST testing accuracy: {standard_acc}")
    print(f"Dropout Net MNIST testing accuracy: {dropout_acc}")
