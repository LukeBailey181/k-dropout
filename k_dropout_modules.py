import time
import numpy as np
import torch
from torch import nn, Tensor


class StochasticKDropout(nn.Module):
    r"""
    Module for k-dropout, where each dropout mask is used for k consecutive steps.
    Input activations should be of shape (batch_size * d), meaning batch dimension
    should be first.
    Arguments:
        k: number of steps to use the same mask.
        p: probability of an element to be zeroed. Default: 0.5
        batch_mask_share: If true, then each activation in the input batched is masked
            the same. If false, each activation has its own mask generated.
    """

    def __init__(self, k: int, p: float = 0.5, batch_mask_share=False):
        super(StochasticKDropout, self).__init__()
        self.k = k
        self.p = p
        self.batch_mask_share = batch_mask_share

        self.uses = 0
        self.seed = torch.Generator().seed()

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            g = torch.Generator(device=x.device)
            if self.uses % self.k == 0:  # update mask seed every k steps
                self.seed = g.seed()
            else:
                g.manual_seed(self.seed)
            self.uses += 1

            if self.batch_mask_share:
                # Share same mask across batch
                batch_size, d = x.shape
                single_mask = torch.rand((d), device=x.device, generator=g) >= self.p
                mask = single_mask.repeat(batch_size, 1)
            else:
                # Use a new mask for each activation
                mask = torch.rand(x.shape, device=x.device, generator=g) >= self.p

            return (1.0 / (1 - self.p)) * (mask * x)  # mask and scale
        return x

    def extra_repr(self) -> str:
        return f"p={self.p}, k={self.k}"


class PoolKDropout(nn.Module):
    r"""
    Module for the pool variant of k-dropout where a pool of n_masks masks are generated
    and at each training step a mask is randomly selected from the pool. n_masks is
    simply a different way of parameterizing k, as setting n_masks = total_steps / k
    means each mask will be used on average k times.
    Arguments:
        n_masks: number of masks in the pool.
        p: probability of an element to be zeroed. Default: 0.5
    """

    def __init__(self, n_masks: int, p: float = 0.5, batch_mask_share=False):
        super(PoolKDropout, self).__init__()
        self.n_masks = n_masks
        self.p = p
        self.batch_mask_share = batch_mask_share

        g = torch.Generator()
        self.mask_seeds = [g.seed() for _ in range(n_masks)]

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            seed_index = torch.randint(high=self.n_masks, size=(1,)).item()
            g = torch.Generator(device=x.device)
            g.manual_seed(self.mask_seeds[seed_index])

            if self.batch_mask_share:
                # Share same mask across batch
                batch_size, d = x.shape
                single_mask = torch.rand((d), device=x.device, generator=g) >= self.p
                mask = single_mask.repeat(batch_size, 1)
            else:
                # Use a new mask for each activation
                mask = torch.rand(x.shape, device=x.device, generator=g) >= self.p

            return (1.0 / (1 - self.p)) * (mask * x)  # mask and scale
        return x

    def extra_repr(self) -> str:
        return f"p={self.p}, n_masks={self.n_masks}"


'''
class RRKDropout(nn.Module):
    r"""
    Module for the round-robin variant of k-dropout where a pool of n_masks masks are
    generated then in training rotated and used for k consecutive steps each.
    Arguments:
        n_masks: number of masks in the pool.
        k: number of steps to use the same mask.
        p: probability of an element to be zeroed. Default: 0.5
    """

    def __init__(self, n_masks: int, k: int, p: float=0.5):
        super(RRKDropout, self).__init__()
        self.n_masks = n_masks
        self.k = k
        self.p = p
        self.uses = 0
        self.generator = torch.Generator()
        self.mask_seeds = [self.generator.seed() for _ in range(n_masks)]
        self.mask_idx = -1

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            if self.uses % self.k == 0:  # rotate mask every k steps
                self.mask_index = (self.mask_idx + 1) % self.n_masks
            self.generator.manual_seed(self.mask_seeds[self.mask_index])
            self.uses += 1

            mask = (torch.rand(x.shape, generator=self.generator) > self.p).to(x.device)
            return mask * x * (1.0 / (1 - self.p))  # mask and scale
        return x

    def extra_repr(self) -> str:
        return f'p={self.p}, n_masks={self.n_masks}, k={self.k}'
'''


if __name__ == "__main__":
    trials = 100
    array_size = 1_000_000
    devices = ["cpu"] + ["cuda"] if torch.cuda.is_available() else []
    rtol = 0.01

    # test correctness
    # stochastic k-dropout
    ps = [0, 0.25, 0.5, 0.75]
    ks = [1, 2, 3, 5]
    for device in devices:
        for k in ks:
            for p in ps:
                skd = StochasticKDropout(k=k, p=p)
                print(f"StochasticKDropout: device={device} k={k} p={p}")
                for t in range(trials):
                    x = torch.ones((array_size,), device=device)
                    out = skd(x).cpu()
                    p_out = 1 - ((out != 0).sum().item() / array_size)
                    if t % k == 0:
                        mask = out != 0

                    assert np.isclose(p, p_out, rtol=rtol)  # p correct
                    assert np.isclose(out.max().item(), 1 / (1 - p))  # scaling correct

    # pool k-dropout
    ps = [0, 0.25, 0.5, 0.75]
    n_masks = [1, 2, 3, 5]
    for device in devices:
        for n_mask in n_masks:
            for p in ps:
                skd = PoolKDropout(n_masks=n_mask, p=p)
                print(f"PoolKDropout: device={device} n_masks={n_mask} p={p}")
                masks = []
                for t in range(trials):
                    x = torch.ones((array_size,), device=device)
                    out = skd(x).cpu()
                    p_out = 1 - ((out != 0).sum().item() / array_size)
                    mask = out != 0
                    for m in masks:
                        if (m == mask).all().item():
                            break
                    else:
                        masks.append(mask)

                    assert np.isclose(p, p_out, rtol=rtol)  # p correct
                    assert np.isclose(out.max().item(), 1 / (1 - p))  # scaling correct
                # mask pool correct
                if p == 0:
                    assert len(masks) == 1
                else:
                    assert len(masks) == n_mask

    # test performance
    # NOTE: we expect nn.Dropout to outperform our layer, we just need this gap
    #       to not be worse on the gpu
    input_size = 100_000
    n_trials = 10_000

    layers = [
        nn.Dropout(p=0.5),
        StochasticKDropout(k=10, p=0.5),
        PoolKDropout(n_masks=100, p=0.5),
    ]

    for device in devices:
        input = torch.randn((input_size,), device=device)
        for layer in layers:
            start_time = time.perf_counter()

            for _ in range(n_trials):
                out = layer(input)

            print(f"{layer}, device={device}: {time.perf_counter() - start_time:.2f}s")
