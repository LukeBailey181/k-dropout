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
        m: number of masks to use per batch, defaults to 0 which will set m=batch_size for
            any input.
    """
    def __init__(self, k: int, p: float = 0.5, m=-1):
        super(SequentialKDropout, self).__init__()
        self.k = k
        self.p = p
        self.m = m

        self.uses = 0
        self.seed = torch.Generator().seed()

    def forward(self, x: Tensor) -> Tensor:

        if self.m == -1:
            self.m = x.shape[0]

        # Check m divides batch size
        batch_size, d = x.shape
        if batch_size % self.m != 0:
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

            mask_len = int(batch_size / self.m)
            mask_block = torch.rand((self.m, d), device=x.device, generator=g) >= self.p
            mask = mask_block.repeat(mask_len, 1)

            return (1.0 / (1 - self.p)) * (mask * x)  # mask and scale

        return x

    def extra_repr(self) -> str:
        return f"p={self.p}, k={self.k}, m={self.m}"


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

    def __init__(self, n_masks: int, p: float = 0.5, m=-1):
        super(PoolKDropout, self).__init__()
        self.n_masks = n_masks
        self.p = p
        self.m = m

        g = torch.Generator()
        self.mask_seeds = [g.seed() for _ in range(n_masks)]

    def forward(self, x: Tensor) -> Tensor:

        if self.m == -1:
            self.m = x.shape[0]

        # Check m divides batch size
        batch_size, d = x.shape
        if batch_size % self.m != 0:
            raise ValueError(
                f"m value of {self.m} does not divide batch_size of value {batch_size}"
            )

        if self.training:
            g = torch.Generator(device=x.device)
            seed_idxs = torch.randint(high=self.n_masks, size=(self.m,)) 
            gen_seeds = [self.mask_seeds[i] for i in seed_idxs]
            
            masks = [] 
            for seed in gen_seeds:
                g.manual_seed(seed) 
                masks.append(torch.rand(d, device=x.device, generator=g) >= self.p)

            mask_block = torch.stack(masks)

            mask_len = int(batch_size / self.m)
            mask = mask_block.repeat(mask_len, 1)

            return (1.0 / (1 - self.p)) * (mask * x)  # mask and scale
        return x

    def extra_repr(self) -> str:
        return f"p={self.p}, n_masks={self.n_masks}, m={self.m}"


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
