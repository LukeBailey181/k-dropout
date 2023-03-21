import time

import torch
import torch.nn as nn
import numpy as np

from k_dropout.modules import SequentialKDropout, PoolKDropout


if __name__ == "__main__":
    trials = 100
    array_size = 1_000_000
    devices = ["cpu"] + ["cuda"] if torch.cuda.is_available() else []
    rtol = 0.01

    # test correctness
    # sequential k-dropout
    ps = [0, 0.25, 0.5, 0.75]
    ks = [1, 2, 3, 5]
    for device in devices:
        for k in ks:
            for p in ps:
                skd = SequentialKDropout(k=k, p=p)
                print(f"SequentialKDropout: device={device} k={k} p={p}")
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

    # TODO: test the batch mask share parameter

    # test performance
    # NOTE: we expect nn.Dropout to outperform our layer, we just need this gap
    #       to not be worse on the gpu
    input_size = 100_000
    n_trials = 10_000

    layers = [
        nn.Dropout(p=0.5),
        SequentialKDropout(k=10, p=0.5),
        PoolKDropout(n_masks=100, p=0.5),
    ]

    for device in devices:
        input = torch.randn((input_size,), device=device)
        for layer in layers:
            start_time = time.perf_counter()

            for _ in range(n_trials):
                out = layer(input)

            print(f"{layer}, device={device}: {time.perf_counter() - start_time:.2f}s")
