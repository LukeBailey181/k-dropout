from torch import nn
from typing import Dict, Optional, Any

from k_dropout.modules import SequentialKDropout, PoolKDropout


# TODO: add option for dropout on the input layer
def make_net(
    input_size: int,
    output_size: int,
    hidden_size: int,
    n_hidden: int,
    dropout_layer: Optional[Any] = None,
    dropout_kwargs: Dict[str, Any] = {},
    input_dropout_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Helper function for making NNs"""

    input_dropout = []
    if input_dropout_kwargs is not None and dropout_layer is not None:
        input_dropout = [dropout_layer(**input_dropout_kwargs)]

    input = input_dropout + [
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
    ]

    if dropout_layer is not None:
        input.append(dropout_layer(**dropout_kwargs))

    hidden = []
    for _ in range(n_hidden):
        hidden.append(nn.Linear(hidden_size, hidden_size))
        hidden.append(nn.ReLU())
        if dropout_layer is not None:
            hidden.append(dropout_layer(**dropout_kwargs))

    output = [nn.Linear(hidden_size, output_size)]

    return nn.Sequential(*input, *hidden, *output)


def make_pt_dropoout_net(
    input_dim: int = 784,
    num_classes: int = 10,
    hidden_units: int = 100,
    hidden_layers: int = 2,
    p: float = 0.5,
) -> nn.Module:
    """Return a NN that uses standard pytorch dropout"""

    return make_net(
        input_dim,
        num_classes,
        hidden_units,
        hidden_layers,
        dropout_layer=nn.Dropout,
        dropout_kwargs={"p": p},
    )


def make_skd_net(
    input_dim: int = 784,
    num_classes: int = 10,
    hidden_units: int = 100,
    hidden_layers: int = 2,
    k: int = 1,
    p: float = 0.5,
    m: int = -1,
) -> nn.Module:
    """Return a NN that uses SequentialKDropout"""

    return make_net(
        input_size=input_dim,
        output_size=num_classes,
        hidden_size=hidden_units,
        n_hidden=hidden_layers,
        dropout_layer=SequentialKDropout,
        dropout_kwargs={"p": p, "k": k, "m": m},
    )


def make_pool_kd_net(
    input_dim: int = 784,
    num_classes: int = 10,
    hidden_units: int = 100,
    hidden_layers: int = 2,
    pool_size: int = 5,
    p: float = 0.5,
    m: int = -1,
) -> nn.Module:
    """Return a NN that uses PoolKDropout"""

    return make_net(
        input_size=input_dim,
        output_size=num_classes,
        hidden_size=hidden_units,
        n_hidden=hidden_layers,
        dropout_layer=PoolKDropout,
        dropout_kwargs={
            "p": p,
            "pool_size": pool_size,
            "m": m,
            "cache_masks": True,
            "input_dim": hidden_units,
        },
    )


def make_standard_net(
    input_dim: int = 784,
    num_classes: int = 10,
    hidden_units: int = 100,
    hidden_layers: int = 2,
) -> nn.Module:
    """Return a NN without dropout"""

    return make_net(
        input_dim, num_classes, hidden_units, hidden_layers, dropout_layer=None
    )


if __name__ == "__main__":
    standard_net = make_standard_net()
    print("STANDARD NET:")
    print(standard_net)

    pt_dropout_net = make_pt_dropoout_net()
    print("PT DROPOUT NET:")
    print(pt_dropout_net)

    skd_net = make_skd_net()
    print("SEQUENTIAL K DROPOUT NET:")
    print(skd_net)

    pool_kd_net = make_pool_kd_net()
    print("POOL K DROPOUT NET:")
    print(pool_kd_net)