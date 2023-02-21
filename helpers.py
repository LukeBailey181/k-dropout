import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt

#DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = "mps"

class StandardNet(nn.Module):
    """ 
    Standard MLP implementation for testing
    """

    def __init__(self, num_classes=2, input_dim=2, hidden_units=100):
        super(StandardNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)  
        self.fc3 = nn.Linear(hidden_units, hidden_units)  
        self.fc4 = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

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
            
            X,y = data 
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            output = net(X)
            loss = criterion(output, y)
            total_loss += loss.item()
            total_correct += (output.argmax(dim=1) == y).sum().item()
            total_examples += len(y)

    return total_loss, total_correct / total_examples
            

def train_net(epochs, net, trainset, lr=0.005):
    """"
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
        torchvision.datasets.MNIST('./datasets/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,)
                                    ),
                                    torchvision.transforms.Lambda(lambda x: torch.flatten(x))
                                ])),
        batch_size=train_batch_size, 
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./datasets/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,)
                                    ),
                                    torchvision.transforms.Lambda(lambda x: torch.flatten(x))
                                    ])),
        batch_size=test_batch_size, 
        shuffle=True
    )

    return train_loader, test_loader

if __name__ == "__main__":

    # Example usage
    train_loader, test_loader = get_mnist()
    net = StandardNet(num_classes=10, input_dim=784)
    train_net(10, net, train_loader)

    _, acc = test_net(net, test_loader)
    print(f"MNIST testing accuracy: {acc}")
