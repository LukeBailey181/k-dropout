import torch
from torch import nn, Tensor
from typing import Optional


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

        self.manual_seed = 0
        self.use_manual_seed = False

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"Input must be of shape (batch_size, d), got {x.shape}")
        batch_size, d = x.shape

        # Check m divides batch size
        if self.m > 0 and batch_size % self.m != 0:
            raise ValueError(
                f"m value of {self.m} does not divide batch_size of value {batch_size}"
            )

        if self.training or self.use_manual_seed:  # when using manual seed, always mask
            batch_mask = self.get_mask()
            return (1.0 / (1 - self.p)) * (batch_mask * x)  # mask and scale

        return x

    def get_mask(self, increment_uses: bool = True) -> torch.Tensor:
        g = torch.Generator(device=x.device)

        if self.use_manual_seed:
            g.manual_seed(self.manual_seed)
        else:
            if self.uses % self.k == 0:  # update mask seed every k steps
                self.seed = g.seed()
            else:
                g.manual_seed(self.seed)
            self.uses += increment_uses

        masks_per_batch = self.m if self.m > 0 else batch_size
        mask_n_repeats = batch_size // masks_per_batch
        mask_block = (
            torch.rand((masks_per_batch, d), device=x.device, generator=g) >= self.p
        )
        batch_mask = mask_block.repeat(mask_n_repeats, 1)
        return batch_mask

    def extra_repr(self) -> str:
        if self.use_manual_seed:
            return f"p={self.p}, k={self.k}, m={self.m}, manual_seed={self.manual_seed}"
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
        cache_masks: If true, then the mask pool is generated once on initialization.
            This requires that the input dimension is fixed.
        input_dim: If cache_masks is true, then this is the fixed input dimension.
    """

    def __init__(
        self,
        pool_size: int,
        p: float = 0.5,
        m: int = -1,
        cache_masks: bool = False,
        input_dim: Optional[int] = None,
        sync_over_model: bool = False,
    ):
        super(PoolKDropout, self).__init__()
        self.pool_size = pool_size
        self.p = p
        self.m = m
        self.cache_masks = cache_masks
        self.input_dim = input_dim
        self.frozen_mask_idx = None
        self.num_training_passes = 0
        self.sync_over_model = sync_over_model

        if self.cache_masks:
            if self.input_dim is None:
                raise ValueError(
                    "If cache_masks is true, then input_dim must be specified."
                )
            g = torch.Generator()
            mask_pool = torch.rand((pool_size, self.input_dim), generator=g) >= self.p
            self.mask_pool = nn.Parameter(mask_pool, requires_grad=False)
        else:
            g = torch.Generator()
            self.mask_seeds = [g.seed() for _ in range(pool_size)]

    def freeze_mask(self, mask_idx: int) -> None:
        """Freeze mask to be one corresponding to mask_idx for training
        and inference
        """
        if mask_idx >= self.pool_size:
            raise ValueError(
                f"mask_idx value of {mask_idx} not valid for pool size of {self.pool_size}"
            )
        self.frozen_mask_idx = mask_idx

    def unfreeze_mask(self) -> None:
        self.frozen_mask_idx = None

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"Input must be of shape (batch_size, d), got {x.shape}")
        batch_size, d = x.shape

        if self.frozen_mask_idx is not None:
            # Use frozen_mask_idx if in training or not

            if self.cache_masks:
                mask = self.mask_pool[self.frozen_mask_idx]
            else:
                g = torch.Generator(device=x.device)
                gen_seed = self.mask_seeds[self.frozen_mask_idx]
                g.manual_seed(gen_seed)
                mask = torch.rand(d, device=x.device, generator=g) >= self.p

            batch_mask = mask.repeat(batch_size, 1)
            return (1.0 / (1 - self.p)) * (batch_mask * x)  # mask and scale

        # Check m divides batch size
        if self.m > 0 and batch_size % self.m != 0:
            raise ValueError(
                f"m value of {self.m} does not divide batch_size of value {batch_size}"
            )

        if self.training:
            self.num_training_passes += 1

            # sample mask indices
            g = torch.Generator(device=x.device)
            masks_per_batch = self.m if self.m > 0 else batch_size

            if self.sync_over_model:
                g.manual_seed(self.num_training_passes)
                seed_idxs = torch.randint(
                    high=self.pool_size,
                    size=(masks_per_batch,),
                    device=x.device,
                    generator=g,
                )
            else:
                seed_idxs = torch.randint(
                    high=self.pool_size,
                    size=(masks_per_batch,),
                    device=x.device,
                )

            # generate batch mask
            if self.cache_masks:
                mask_block = self.mask_pool[seed_idxs]
            else:
                gen_seeds = [self.mask_seeds[i] for i in seed_idxs]
                mask_block = torch.empty((masks_per_batch, d), device=x.device)
                for i, seed in enumerate(gen_seeds):
                    g.manual_seed(seed)
                    mask_block[i] = (
                        torch.rand(d, device=x.device, generator=g) >= self.p
                    )

            mask_n_repeats = batch_size // masks_per_batch
            batch_mask = mask_block.repeat(mask_n_repeats, 1)

            return (1.0 / (1 - self.p)) * (batch_mask * x)  # mask and scale
        return x

    def extra_repr(self) -> str:
        return f"p={self.p}, pool_size={self.pool_size}, m={self.m}, cache_masks={self.cache_masks}, input_dim={self.input_dim}, sync_over_model={self.sync_over_model}"


if __name__ == "__main__":
    # Test
    batch_size = 8
    features = 4
    dummy_batch = torch.rand((batch_size, features))

    for cache_masks in [True, False]:
        print(f"cache_masks := {cache_masks}")

        pool_dropout = PoolKDropout(
            pool_size=2,
            m=batch_size,
            p=0.5,
            cache_masks=cache_masks,
            input_dim=features,
        )
        pool_dropout.freeze_mask(0)
        pool_dropout.eval()

        print("Input:")
        print(dummy_batch)
        print("Frozen mask output:")
        print(pool_dropout(dummy_batch))

        pool_dropout.unfreeze_mask()
        pool_dropout.train()

        print("Unfrozen mask train output:")
        print(pool_dropout(dummy_batch))
