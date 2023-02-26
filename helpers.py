import time
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt

from networks import make_standard_net, make_skd_net


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def test_net(net, dataset):
    """
    Evaulates inputted net on inputted dataset
    """

    criterion = nn.CrossEntropyLoss()
    net.to(DEVICE)
    net.eval()
    total_loss = total_correct = total_examples = 0
    with torch.no_grad():
        for data in dataset:

            X, y = data
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            output = net(X)
            loss = criterion(output, y)
            total_loss += loss.item()
            total_correct += (output.argmax(dim=1) == y).sum().item()
            total_examples += len(y)

    return total_loss, total_correct / total_examples


def train_net(epochs, net, trainset, lr=0.005, plot=False):
    """ "
    Trains inputted net using provided trainset.
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)

    losses = []
    epoch_losses = []

    net.train()
    net.to(DEVICE)
    for epoch in range(epochs):
        epoch_loss = 0
        if epoch % 1 == 0:
            print(f"Epoch {epoch}")
        for data in trainset:
            X, y = data
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            net.zero_grad()
            output = net(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            epoch_loss += loss.item()
        epoch_losses.append(epoch_loss)

    if plot:
        plt.plot([i for i in range(len(losses[10:]))], losses[10:])
        plt.title("Training Loss")
        plt.xlabel("Batch")
        plt.show()

        plt.plot([i for i in range(len(epoch_losses))], epoch_losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.show()


def get_mnist(train_batch_size=64, test_batch_size=1000):
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
