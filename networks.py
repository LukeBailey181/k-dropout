import torch
from torch import nn

from k_dropout_modules import StochasticKDropout


def make_skd_net(input_dim=784, num_classes=10, hidden_units=100, k=1, p=0.5):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_units),
        nn.ReLU(),
        StochasticKDropout(k, p),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(),
        StochasticKDropout(k, p),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(),
        StochasticKDropout(k, p),
        nn.Linear(hidden_units, num_classes),
    )


def make_standard_net(input_dim=784, num_classes=10, hidden_units=100):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, num_classes),
    )
