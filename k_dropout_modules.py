import torch
from torch import nn


class StochasticKDropout(nn.Module):
    r"""
    Module for k-dropout, where each dropout mask is used for k consecutive steps.
    Arguments:
        k: number of steps to use the same mask.
        p: probability of an element to be zeroed. Default: 0.5
    """

    def __init__(self, k, p=0.5):
        super(StochasticKDropout, self).__init__()
        self.k = k
        self.p = p
        self.uses = 0
        self.generator = torch.Generator()
        self.seed = self.generator.initial_seed()

    def forward(self, x):
        if self.training:
            if self.uses % self.k == 0:  # update mask seed every k steps
                self.seed = self.generator.seed()
            else:
                self.generator.manual_seed(self.seed)
            self.uses += 1

            mask = (torch.rand(x.shape, generator=self.generator) > self.p).to(x.device)
            return mask * x * (1.0 / (1 - self.p))  # mask and scale
        return x


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

    def __init__(self, n_masks, p=0.5):
        super(PoolKDropout, self).__init__()
        self.n_masks = n_masks
        self.p = p
        self.generator = torch.Generator()
        self.mask_seeds = [self.generator.seed() for _ in range(n_masks)]

    def forward(self, x):
        if self.training:
            seed_index = torch.randint(high=self.n_masks, size=(1,)).item()
            self.generator.manual_seed(self.mask_seeds[seed_index])

            mask = (torch.rand(x.shape, generator=self.generator) > self.p).to(x.device)
            return mask * x * (1.0 / (1 - self.p))  # mask and scale
        return x


class RRKDropout(nn.Linear):
    r"""
    Module for the round-robin variant of k-dropout where a pool of n_masks masks are
    generated then in training rotated and used for k consecutive steps each.
    Arguments:
        n_masks: number of masks in the pool.
        k: number of steps to use the same mask.
        p: probability of an element to be zeroed. Default: 0.5
    """

    def __init__(self, n_masks, k, p=0.5):
        super(RRKDropout, self).__init__()
        self.n_masks = n_masks
        self.k = k
        self.p = p
        self.uses = 0
        self.generator = torch.Generator()
        self.mask_seeds = [self.generator.seed() for _ in range(n_masks)]
        self.mask_idx = -1

    def forward(self, x):
        if self.training:
            if self.uses % self.k == 0:  # rotate mask every k steps
                self.mask_index = (self.mask_idx + 1) % self.n_masks
            self.generator.manual_seed(self.mask_seeds[self.mask_index])
            self.uses += 1

            mask = (torch.rand(x.shape, generator=self.generator) > self.p).to(x.device)
            return mask * x * (1.0 / (1 - self.p))  # mask and scale
        return x
