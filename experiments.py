import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict

from helpers import train_net, test_net, get_mnist
from networks import make_standard_net, make_skd_net, make_pt_dropoout_net


def test_k_values_on_MNIST(k_vals, p=0.5):

    # Train standard net for baseline data
    train_loader, test_loader = get_mnist()
    standard_net = make_standard_net(num_classes=10, input_dim=784)
    train_net(20, standard_net, train_loader)
    _, standard_acc = test_net(standard_net, test_loader)

    # Train kdropout net
    dropout_accs = []
    for k in k_vals:
        train_loader, test_loader = get_mnist()
        dropout_net = make_skd_net(num_classes=10, input_dim=784, p=0.5, k=k)
        train_net(20, dropout_net, train_loader)

        _, acc = test_net(dropout_net, test_loader)
        dropout_accs.append(acc)

    print(f"Standard acc: {standard_acc}")
    print(f"KDropout accs: {dropout_accs}")


def find_performant_dropout_net(
    hidden_layers=[2, 5], hidden_units=[100, 200, 500], p=0.5, repeats=5
):
    """
    Trains multiple standard and pytorch dropout nets to find an
    architecture in which regular dropout performs better than a standard
    networks
    """

    results = defaultdict(list)
    for num_hl in hidden_layers:
        for num_hu in hidden_units:
            for _ in range(repeats):
                train_loader, test_loader = get_mnist()

                # Train and test standard net
                standard_net = make_standard_net(
                    num_classes=10,
                    input_dim=784,
                    hidden_units=num_hu,
                    hidden_layers=num_hl,
                )
                train_net(20, standard_net, train_loader)
                _, standard_acc = test_net(standard_net, test_loader)

                results[("standard", num_hl, num_hu)].append(standard_acc)

                # Train and test dropout net
                dropout_net = make_pt_dropoout_net(
                    num_classes=10,
                    input_dim=784,
                    hidden_units=num_hu,
                    hidden_layers=num_hl,
                    p=p,
                )
                train_net(20, dropout_net, train_loader)
                _, dropout_acc = test_net(standard_net, test_loader)

                results[("dropout", num_hl, num_hu)].append(dropout_acc)

    print(dict(results))


if __name__ == "__main__":

    # TODO RETURN THIS BACK
    # test_k_values_on_MNIST(k_vals=[1,5,10,50, 100, 200, 300, 500, 800])
    # test_k_values_on_MNIST(k_vals=[1,5,10])

    find_performant_dropout_net(
        hidden_layers=[2, 5], hidden_units=[100, 200, 500], p=0.5, repeats=5
    )
