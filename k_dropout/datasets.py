import torch
from torchvision import transforms, datasets


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

DATASET_ROOT = "./datasets/"


def process_dataset(dataset, device=DEVICE):
    """
    Iterates through dataset, applying transforms associated with dataloader and
    moving tensors onto device
    """

    processed_data = []
    for X, y in dataset:
        processed_data.append((X.to(device), y.to(device)))
    return processed_data


def get_mnist(train_batch_size=64, test_batch_size=1000, num_workers=2):
    """
    Download MNIST into ./datasets/ directory and return dataloaders containing
    MNIST. If ./datasets/ directory doesn't exist then it is made.

    Keyword arguments:
    train_batch_size -- size of batches in returned train set dataloader
    test_batch_size -- size of batches in returned test set dataloader
    """

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # to mean 0, std 1
            transforms.Lambda(lambda x: torch.flatten(x)),
            ])

    train_set = datasets.MNIST(
            root=DATASET_ROOT, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(
            root=DATASET_ROOT, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=train_batch_size, shuffle=True, drop_last=True,
            pin_memory=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=test_batch_size, shuffle=False, drop_last=True,
            pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader


def get_cifar10(train_batch_size=4, test_batch_size=4, num_workers=2):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # to [-1, 1]
            transforms.Lambda(lambda x: torch.flatten(x)),
            ])

    train_set = datasets.CIFAR10(
        root=DATASET_ROOT, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(
        root=DATASET_ROOT, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=train_batch_size, shuffle=True, drop_last=True,
        pin_memory=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False, drop_last=True,
        pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader
