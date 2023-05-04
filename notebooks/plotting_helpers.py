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
    n_rows = round(n_subnets / n_cols + 0.49)
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
        if type == 'mask':
            ax.set_title(f'Dropout Subnet {i}')
        else:
            ax.set_title(f'Random Subnet {i}')

        if type == 'mask':
            training_start = i * epochs_per_mask
            training_end = (i + 1) * epochs_per_mask
            ax.axvspan(training_start, training_end, color='red', alpha=.2)
        if mask_lines:
            for j in range(n_masks + 1):
                ax.axvline(x=epochs_per_mask * j, color='black',
                           linestyle='--', alpha=.5, linewidth=.5)

        if i % n_cols == 0:
            ax.set_ylabel('Test Accuracy')

        if i // n_cols == n_rows - 1:
            ax.set_xlabel('Epoch')

    # fig.suptitle(f'Sequential k-dropout k={k}, m={m} {type} subnets\n', fontsize=24)
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
    # n_random_subnets = run_config['n_random_subnets']

    n_masks = epochs * BATCHES_PER_EPOCH // k
    # epochs_per_mask = epochs / n_masks

    mask_cols = [f'test_{metric}_mask_{i}' for i in range(n_masks)]
    # random_cols = [f'test_{metric}_random_{i}' for i in range(n_random_subnets)]

    summary = get_run_summary(run_id)

    mask_values = [summary[c] for c in mask_cols]
    # random_values = [summary[c] for c in random_cols]
    plt.scatter(range(n_masks), mask_values)
    # plt.scatter(range(n_masks), mask_values, label='dropout subnets')
    # plt.scatter(np.linspace(0, n_masks - 1, n_random_subnets), random_values, color='r', alpha=.5, label='random subnets')
    # plt.axhline(y=np.array(random_values).mean(), color='r', alpha=.5, label='random subnets (mean)')

    plt.title(f'Sequential k-dropout k={k}, m={m}')
    plt.xlabel('Mask Index')
    plt.ylabel(f'Test {"Accuracy" if metric == "acc" else "Loss"} at End of Training')
    # plt.legend()
    plt.show()


def plot_subnet_boxplots(
        rid_seq, rid_std, rid_no, metric='acc', offset_full_ticks=False,
        figsize=(14, 5), width_ratios=(2, 1)):
    METRICS = ('acc', 'loss')
    if metric not in METRICS:
        raise ValueError(f'metric must be one of {METRICS}')
    if isinstance(rid_seq, str):
        rid_seq = [rid_seq]

    rconf_seq = [get_run_config(rid) for rid in rid_seq]
    rconf_std = get_run_config(rid_std)
    rconf_no = get_run_config(rid_no)
    summary_seq = [get_run_summary(rid) for rid in rid_seq]
    summary_std = get_run_summary(rid_std)
    summary_no = get_run_summary(rid_no)

    n_seq_random_subnets = [rconf['n_random_subnets'] for rconf in rconf_seq]
    n_std_random_subnets = rconf_std['n_random_subnets']
    n_no_random_subnets = rconf_no['n_random_subnets']
    ks = [rconf['k'] for rconf in rconf_seq]
    epochs = [rconf['epochs'] for rconf in rconf_seq]
    n_masks = [e * BATCHES_PER_EPOCH // k for e, k in zip(epochs, ks)]

    mask_cols = [[f'test_{metric}_mask_{i}' for i in range(n)] for n in n_masks]
    rand_seq_cols = [[f'test_{metric}_random_{i}' for i in range(n)] for n in n_seq_random_subnets]
    rand_std_cols = [f'test_{metric}_random_{i}' for i in range(n_std_random_subnets)]
    rand_no_cols = [f'test_{metric}_random_{i}' for i in range(n_no_random_subnets)]

    mask_values = [[ss[c] for c in mc] for mc, ss in zip(mask_cols, summary_seq)]
    rand_seq_values = [[ss[c] for c in rsc] for rsc, ss in zip(rand_seq_cols, summary_seq)]
    rand_std_values = [summary_std[c] for c in rand_std_cols]
    rand_no_values = [summary_no[c] for c in rand_no_cols]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, width_ratios=width_ratios)

    all_values = [rand_no_values, rand_std_values]
    colors = ['C0', 'C1']
    labels = ['no dropout\nrandom subnets', 'standard dropout\nrandom subnets']
    for i, (r, m, k) in enumerate(zip(rand_seq_values, mask_values, ks)):
        all_values.extend([r, m])
        colors.extend(2 * [f'C{i + 2}'])
        labels.extend([
            f'sequential (k={k})\nrandom subnets',
            f'sequential (k={k})\ndropout subnets',
        ])
    bp = axes[0].boxplot(all_values,
                    whis=(0, 100), patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_title('Subnet Performance')
    axes[0].set_ylabel('Test Accuracy at End of Training')
    for tick in axes[0].xaxis.get_major_ticks()[1::2]:
        tick.set_pad(30)
    bottom = axes[0].get_ylim()[0]

    axes[1].set_title('Full Network Performance')
    bars = [summary_no[f'test_{metric}_full'], summary_std[f'test_{metric}_full']]
    bars += [ss[f'test_{metric}_full'] for ss in summary_seq]
    labels = ['no dropout', 'standard dropout']
    labels += [f'sequential (k={k})' for k in ks]
    axes[1].bar(
        range(len(bars)),
        bars,
        color=[f'C{i}' for i in range(len(bars))], width=.5, align='center')
    axes[1].set_xticks(range(len(labels)), labels, fontsize=10)
    if offset_full_ticks:
        for tick in axes[1].xaxis.get_major_ticks()[1::2]:
            tick.set_pad(30)
    axes[1].set_ylim(bottom=bottom)

    fig.tight_layout()