import torch
from torch import nn, Tensor


class SequentialKDropout(nn.Module):
    r"""
    Module for k-dropout, where each dropout mask is used for k consecutive steps.
    Input activations should be of shape (batch_size * d), meaning batch dimension
    should be first.
    Arguments:
        k: number of steps to use the same mask.
        p: probability of an element to be zeroed. Default: 0.5
        batch_mask_share: If true, then each activation in the input batched is masked
            the same. If false, each activation has its own mask generated.
        m: number of masks to use per batch, defaults to -1 which will set m=batch_size
            for any input.
    """

    def __init__(self, k: int, p: float = 0.5, m=-1):
        super(SequentialKDropout, self).__init__()
        self.k = k
        self.p = p
        self.m = m

        self.uses = 0
        self.seed = torch.Generator().seed()

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"Input must be of shape (batch_size, d), got {x.shape}")
        batch_size, d = x.shape

        # Check m divides batch size
        if self.m > 0 and batch_size % self.m != 0:
            raise ValueError(
                f"m value of {self.m} does not divide batch_size of value {batch_size}"
            )

        if self.training:
            g = torch.Generator(device=x.device)
            if self.uses % self.k == 0:  # update mask seed every k steps
                self.seed = g.seed()
            else:
                g.manual_seed(self.seed)
            self.uses += 1

            masks_per_batch = self.m if self.m > 0 else batch_size
            mask_n_repeats = batch_size // masks_per_batch
            mask_block = (
                torch.rand((masks_per_batch, d), device=x.device, generator=g) >= self.p
            )
            batch_mask = mask_block.repeat(mask_n_repeats, 1)

            return (1.0 / (1 - self.p)) * (batch_mask * x)  # mask and scale

        return x

    def extra_repr(self) -> str:
        return f"p={self.p}, k={self.k}, m={self.m}"


class PoolKDropout(nn.Module):
    r"""
    Module for the pool variant of k-dropout where a pool of pool_size masks are generated
    and at each training step a mask is randomly selected from the pool. pool_size is
    simply a different way of parameterizing k, as setting pool_size = total_steps / k
    means each mask will be used on average k times.
    Arguments:
        pool_size: number of masks in the pool.
        p: probability of an element to be zeroed. Default: 0.5
        m: number of masks to use per batch, defaults to -1 which will set m=batch_size
            for any input.
    """

    def __init__(self, pool_size: int, p: float = 0.5, m=-1):
        super(PoolKDropout, self).__init__()
        self.pool_size = pool_size
        self.p = p
        self.m = m

        g = torch.Generator()
        self.mask_seeds = [g.seed() for _ in range(pool_size)]

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"Input must be of shape (batch_size, d), got {x.shape}")
        batch_size, d = x.shape

        # Check m divides batch size
        if self.m > 0 and batch_size % self.m != 0:
            raise ValueError(
                f"m value of {self.m} does not divide batch_size of value {batch_size}"
            )

        if self.training:
            # TODO: improve performance
            g = torch.Generator(device=x.device)
            masks_per_batch = self.m if self.m > 0 else batch_size
            seed_idxs = torch.randint(high=self.pool_size, size=(masks_per_batch,))
            gen_seeds = [self.mask_seeds[i] for i in seed_idxs]

            mask_block = torch.empty((masks_per_batch, d), device=x.device)
            for i, seed in enumerate(gen_seeds):
                g.manual_seed(seed)
                mask_block[i] = torch.rand(d, device=x.device, generator=g) >= self.p

            mask_n_repeats = batch_size // masks_per_batch
            batch_mask = mask_block.repeat(mask_n_repeats, 1)

            return (1.0 / (1 - self.p)) * (batch_mask * x)  # mask and scale
        return x

    def extra_repr(self) -> str:
        return f"p={self.p}, pool_size={self.pool_size}, m={self.m}"
