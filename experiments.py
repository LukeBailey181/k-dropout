import torch
from torch import nn
import torch.nn.functional as F
from helpers import train_net, test_net, get_mnist, StandardNet
from k_dropout_modules import StochasticKDropoutNet

def test_k_values_on_MNIST(k_vals, p=0.5):

    # Train standard net for baseline data
    train_loader, test_loader = get_mnist()
    standard_net = StandardNet(num_classes=10, input_dim=784)
    train_net(20, standard_net, train_loader)
    _, standard_acc = test_net(standard_net, test_loader)

    # Train kdropout net
    dropout_accs = []
    for k in k_vals:
        train_loader, test_loader = get_mnist()
        dropout_net = StochasticKDropoutNet(num_classes=10, input_dim=784, drop_p=0.5, k=k)
        train_net(20, dropout_net, train_loader)

        _, acc = test_net(dropout_net, test_loader)
        dropout_accs.append(acc)

    print(f"Standard acc: {standard_acc}")
    print(f"KDropout accs: {dropout_accs}")

    
if __name__ == "__main__":
   
    test_k_values_on_MNIST(k_vals=[1,5,10,50, 100, 200, 300, 500, 800])

