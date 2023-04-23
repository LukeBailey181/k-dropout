from torch import nn
import torch
from typing import Dict, Optional, Any
import math

from k_dropout.modules import SequentialKDropout, PoolKDropout

import torch.nn as nn

# from modules import SequentialKDropout, PoolKDropout


class PoolDropoutLensNet(nn.Module):
    def __init__(
        self,
        input_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        hidden_units: Optional[int] = None,
        hidden_layers: Optional[int] = None,
        pool_size: Optional[int] = None,
        p: Optional[float] = None,
        m: Optional[int] = None,
        init_net: Optional[nn.Sequential] = None, 
    ):
        """
        Args:
            input_dim: dataset input dim 
            num_classes: dataset num classes 
            hidden_units: number of hidden units for net
            hidden_layers: number of hidden layers for net
            pool_size: dropout pool size for net
            p: probability of dropout for all dropout layers of net
            m: number of dropout masks per batch
            init_net: if not none, then this is used for self.net instead
                of initializing a new net from the above arguments. This 
                should be an MLP with only pooled dropout layers.
        """

        super().__init__()

        if init_net is not None:
            self.net = init_net
        else:
            self.net = make_pool_kd_net(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_units=hidden_units,
                hidden_layers=hidden_layers,
                pool_size=pool_size,
                p=p,
                m=m,
                sync_over_model=True,
            )

        # Used when sampling random subnets
        self.using_random_masking = False
        self.pooled_dropout_layers = [layer for layer in self.net if isinstance(layer, PoolKDropout)]
        self.p = self.pooled_dropout_layers[-1].p

        return

    def activate_random_masking(self):

        if self.using_random_masking:
            return
        self.using_random_masking = True

        # Create a new random mask and freeze this
        for layer_idx in range(len(self.net)):
            if isinstance(self.net[layer_idx], PoolKDropout):
                self.net[layer_idx] = nn.Dropout(p=self.p)

    def deactivate_random_masking(self):

        if not self.using_random_masking:
            return
        self.using_random_masking = False

        pooled_droout_idx = 0
        for layer_idx in range(len(self.net)):
            if isinstance(self.net[layer_idx], PoolKDropout):
                self.net[layer_idx] = self.pooled_dropout_layers[pooled_droout_idx]
                pooled_droout_idx += 1

    def freeze_mask(self, mask_idx):

        if self.using_random_masking:
            raise RuntimeError("Unable to freeze mask as currently using random masking.")

        for layer in self.net:
            if isinstance(layer, PoolKDropout):
                layer.freeze_mask(mask_idx)

    def unfreeze_mask(self):

        if self.using_random_masking:
            raise RuntimeError("Unable to unfreeze mask as currently using random masking.")

        for layer in self.net:
            if isinstance(layer, PoolKDropout):
                layer.unfreeze_mask()

    @torch.no_grad()
    def reset_weights(self):

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                # Reinit weights using same code as nn.Linear source
                # at https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
                nn.init.kaiming_uniform_(layer.weight.data, a=math.sqrt(5))
                if layer.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(layer.bias, -bound, bound)

    def forward(self, x):
        return self.net(x)


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
    sync_over_model: bool = False,
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
            "sync_over_model": sync_over_model,
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
