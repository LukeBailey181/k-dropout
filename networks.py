from torch import nn
from typing import Dict, Optional, Any

from k_dropout_modules import StochasticKDropout, PoolKDropout


def init_weights(m):
    """He init weights"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")


def _make_net(
    input_dim: int,
    num_classes: int,
    hidden_units: int,
    hidden_layers: int,
    dropout_layer: Optional[Any] = None,
    dropout_kargs: Dict[str, Any] = {},
) -> nn.Module:
    """Helper function for making NNs"""

    input = [
        nn.Linear(input_dim, hidden_units),
        nn.ReLU(),
    ]
    if dropout_layer is not None:
        input.append(dropout_layer(**dropout_kargs))

    hidden = []
    for _ in range(hidden_layers):
        hidden.append(nn.Linear(hidden_units, hidden_units))
        hidden.append(nn.ReLU())
        if dropout_layer is not None:
            hidden.append(dropout_layer(**dropout_kargs))

    output = [nn.Linear(hidden_units, num_classes)]

    return nn.Sequential(*input, *hidden, *output)


def make_pt_dropoout_net(
    input_dim: int = 784,
    num_classes: int = 10,
    hidden_units: int = 100,
    hidden_layers: int = 2,
    p: float = 0.5,
) -> nn.Module:
    """Return a NN that uses standard pytorch dropout"""

    return _make_net(
        input_dim,
        num_classes,
        hidden_units,
        hidden_layers,
        dropout_layer=nn.Dropout,
        dropout_kargs={"p": p},
    )


def make_skd_net(
    input_dim: int = 784,
    num_classes: int = 10,
    hidden_units: int = 100,
    hidden_layers: int = 2,
    k: int = 1,
    p: float = 0.5,
    batch_mask_share: bool = False,
) -> nn.Module:
    """Return a NN that uses StochasticKDropout"""

    return _make_net(
        input_dim,
        num_classes,
        hidden_units,
        hidden_layers,
        dropout_layer=StochasticKDropout,
        dropout_kargs={"p": p, "k": k, "batch_mask_share": batch_mask_share},
    )


def make_pool_kd_net(
    input_dim: int = 784,
    num_classes: int = 10,
    hidden_units: int = 100,
    hidden_layers: int = 2,
    n_masks: int = 5,
    p: float = 0.5,
    batch_mask_share: bool = False,
) -> nn.Module:
    """Return a NN that uses StochasticKDropout"""

    return _make_net(
        input_dim,
        num_classes,
        hidden_units,
        hidden_layers,
        dropout_layer=PoolKDropout,
        dropout_kargs={
            "p": p,
            "n_masks": n_masks,
            "batch_mask_share": batch_mask_share,
        },
    )


def make_standard_net(
    input_dim: int = 784,
    num_classes: int = 10,
    hidden_units: int = 100,
    hidden_layers: int = 2,
) -> nn.Module:
    """Return a NN without dropout"""

    return _make_net(
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
    print("STOCHASTIC K DROPOUT NET:")
    print(skd_net)

    pool_kd_net = make_pool_kd_net()
    print("POOL K DROPOUT NET:")
    print(pool_kd_net)
