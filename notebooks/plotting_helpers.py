import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BATCHES_PER_EPOCH = 97
api = wandb.Api()


def get_sweep_dataframe(sweep_id, project_id='k-dropout'):
    sweep = api.sweep(f'{project_id}/{sweep_id}')

    run_data = []
    for run in sweep.runs:
        run_data.append({
            'run_id': run.id,
            'name': run.name,
            'state': run.state,
            **run.config,
            **run.summary,
        })
    return pd.DataFrame(run_data)


def get_run_config(run_id):
    run = api.run(f'k-dropout/{run_id}')
    return run.config


def get_run_summary(run_id):
    run = api.run(f'k-dropout/{run_id}')
    return run.summary


def get_run_metric(run_id, keys=None):
    run = api.run(f'k-dropout/{run_id}')
    if isinstance(keys, str):
        keys = [keys]
    if keys is None:
        return run.history(keys=keys, samples=1_000_000)
    return run.history(keys=keys, samples=1_000_000)[keys]


def plot_subnet_training(
        run_id, metric='acc', type='mask', limit_subnets=-1, mask_lines=True,
        skip_pretraining_step=True, n_cols=3, col_size=5, row_size=4):
    METRICS = ('acc', 'loss')
    TYPES = ('mask', 'random')
    if metric not in METRICS:
        raise ValueError(f'metric must be one of {METRICS}')
    if type not in TYPES:
        raise ValueError(f'type must be one of {TYPES}')

    run_config = get_run_config(run_id)
    k = run_config['k']
    m = run_config['m']
    epochs = run_config['epochs']
    n_random_subnets = run_config['n_random_subnets']

    n_masks = epochs * BATCHES_PER_EPOCH // k
    epochs_per_mask = epochs / n_masks

    n_subnets = n_masks if type == 'mask' else n_random_subnets
    if limit_subnets > 0:
        n_subnets = min(n_subnets, limit_subnets)
    metric_cols = [f'test_{metric}_{type}_{i}' for i in range(n_subnets)]
    metric_df = get_run_metric(run_id, metric_cols)

    # plot test accuracy for each mask
    n_rows = round(n_subnets / n_cols + 0.5)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(col_size * n_cols, row_size * n_rows), sharey=True)
    axes = axes.reshape((-1, n_cols))  # handle 1 row case

    for i, ax in enumerate(axes.flatten()):
        if i >= n_subnets:
            ax.axis('off')
            continue

        if skip_pretraining_step:
            ax.plot(metric_df[metric_cols[i]][1:])
        else:
            ax.plot(metric_df[metric_cols[i]])
        ax.set_title(f'{type} subnet {i}')

        if type == 'mask':
            training_start = i * epochs_per_mask
            training_end = (i + 1) * epochs_per_mask
            ax.axvspan(training_start, training_end, color='red', alpha=.2)
        if mask_lines:
            for j in range(n_masks + 1):
                ax.axvline(x=epochs_per_mask * j, color='black',
                           linestyle='--', alpha=.5, linewidth=.5)

        if i % n_cols == 0:
            ax.set_ylabel(f'test_{metric}')

    fig.suptitle(f'Sequential k-dropout k={k}, m={m} {type} subnets\n', fontsize=24)
    fig.tight_layout()

    plt.show()


def plot_full_training(run_id, metric='acc', mask_lines=True, skip_pretraining_step=True):
    METRICS = ('acc', 'loss')
    if metric not in METRICS:
        raise ValueError(f'metric must be one of {METRICS}')

    run_config = get_run_config(run_id)
    k = run_config['k']
    m = run_config['m']
    epochs = run_config['epochs']

    n_masks = epochs * BATCHES_PER_EPOCH // k
    epochs_per_mask = epochs / n_masks

    metric_col = f'test_{metric}_full'
    metric_df = get_run_metric(run_id, metric_col)

    if skip_pretraining_step:
        plt.plot(metric_df[metric_col][1:])
    else:
        plt.plot(metric_df[metric_col])

    if mask_lines:
        for j in range(n_masks + 1):
            plt.axvline(x=epochs_per_mask * j, color='black',
                        linestyle='--', alpha=.5, linewidth=.5)

    plt.title(f'Sequential k-dropout k={k}, m={m} full network')
    plt.ylabel(f'test_{metric}')
    plt.show()


def plot_subnet_performance(run_id, metric='acc'):
    METRICS = ('acc', 'loss')
    if metric not in METRICS:
        raise ValueError(f'metric must be one of {METRICS}')

    run_config = get_run_config(run_id)
    k = run_config['k']
    m = run_config['m']
    epochs = run_config['epochs']
    n_random_subnets = run_config['n_random_subnets']

    n_masks = epochs * BATCHES_PER_EPOCH // k
    epochs_per_mask = epochs / n_masks

    mask_cols = [f'test_{metric}_mask_{i}' for i in range(n_masks)]
    random_cols = [f'test_{metric}_random_{i}' for i in range(n_random_subnets)]

    summary = get_run_summary(run_id)

    mask_values = [summary[c] for c in mask_cols]
    random_values = [summary[c] for c in random_cols]
    plt.scatter(range(n_masks), mask_values, label='dropout subnets')
    plt.scatter(np.linspace(0, n_masks - 1, n_random_subnets), random_values, color='r', alpha=.5, label='random subnets')
    plt.axhline(y=np.array(random_values).mean(), color='r', alpha=.5, label='random subnets (mean)')

    plt.title(f'Sequential k-dropout k={k}, m={m}')
    plt.xlabel('mask index')
    plt.ylabel(f'test_{metric}')
    plt.legend()
    plt.show()