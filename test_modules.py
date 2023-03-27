import time
import argparse

import torch
import torch.nn as nn
import numpy as np

from k_dropout.modules import SequentialKDropout, PoolKDropout

TRIALS = 100
ARRAY_SIZE = 1_000_000
RTOL = 0.01

DEVICES = ["cpu"] + ["cuda"] if torch.cuda.is_available() else []

PS = [0, 0.25, 0.5, 0.75]
KS = [1, 2, 3, 5]
POOL_SIZES = [1, 2, 3, 5]

PERF_BATCH_SIZE = 128
PERF_N_TRIALS = 20
PERF_INPUT_SIZE = 100_000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--performance", action="store_true")
    args = parser.parse_args()

    # test correctness
    if args.correctness:
        # sequential k-dropout
        for device in DEVICES:
            for k in KS:
                for p in PS:
                    skd = SequentialKDropout(k=k, p=p)
                    print(f"SequentialKDropout: device={device} k={k} p={p}")
                    for t in range(TRIALS):
                        x = torch.ones(
                            (
                                1,
                                ARRAY_SIZE,
                            ),
                            device=device,
                        )
                        out = skd(x).cpu()
                        p_out = 1 - ((out != 0).sum().item() / ARRAY_SIZE)
                        if t % k == 0:
                            mask = out != 0

                        assert np.isclose(p, p_out, rtol=RTOL)  # p correct
                        assert np.isclose(
                            out.max().item(), 1 / (1 - p)
                        )  # scaling correct

        # pool k-dropout
        for device in DEVICES:
            for pool_size in POOL_SIZES:
                for p in PS:
                    skd = PoolKDropout(pool_size=pool_size, p=p)
                    print(f"PoolKDropout: device={device} pool_size={pool_size} p={p}")
                    masks = []
                    for t in range(TRIALS):
                        x = torch.ones(
                            (
                                1,
                                ARRAY_SIZE,
                            ),
                            device=device,
                        )
                        out = skd(x).cpu()
                        p_out = 1 - ((out != 0).sum().item() / ARRAY_SIZE)
                        mask = out != 0
                        for m in masks:
                            if (m == mask).all().item():
                                break
                        else:
                            masks.append(mask)

                        assert np.isclose(p, p_out, rtol=RTOL)  # p correct
                        assert np.isclose(
                            out.max().item(), 1 / (1 - p)
                        )  # scaling correct
                    # mask pool correct
                    if p == 0:
                        assert len(masks) == 1
                    else:
                        assert len(masks) == pool_size

    # test performance
    # NOTE: we expect nn.Dropout to outperform our layer, we just need this gap
    #       to not be worse on the gpu
    if args.performance:
        layers = [
            nn.Dropout(p=0.5),
            SequentialKDropout(k=10, p=0.5),
            PoolKDropout(pool_size=100, p=0.5),
            PoolKDropout(
                pool_size=100, p=0.5, cache_masks=True, input_dim=PERF_INPUT_SIZE
            ),
        ]

        for device in DEVICES:
            input = torch.randn(
                (
                    PERF_BATCH_SIZE,
                    PERF_INPUT_SIZE,
                ),
                device=device,
            )
            for layer in layers:
                layer = layer.to(device)

                start_time = time.perf_counter()
                for _ in range(PERF_N_TRIALS):
                    out = layer(input)

                print(
                    f"{layer}, device={device}: {time.perf_counter() - start_time:.4f}s"
                )
