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
    """

    # TODO: masks per batch (or just m?), batch_dim: int = 0
    def __init__(self, k: int, p: float = 0.5, batch_mask_share=False):
        super(SequentialKDropout, self).__init__()
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

    # TODO: update batch mask share
    # TODO: rename n_masks to pool_size
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
