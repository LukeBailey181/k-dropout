# CS 229 Final Project: k-Dropout

Dropout is a widely used regularization technique that helps improve the generalization capabilities of deep neural networks. In this paper, we introduce k-dropout, a novel generalization of the standard dropout technique. In k-dropout, instead of resampling dropout masks at every training iteration, masks are reused multiple times according to new $k$ and $m$ tunable hyperparameters. Resuing dropout masks leads to a reduced number of unique subnets being trained compared to traditional dropout. We empirically demonstrate that training a multilayer perceptron (MLP) on CIFAR-10 with as few as 50 distinct subnets using k-dropout yields performance comparable to that of regular dropout. Furthermore, we provide detailed analysis of the trade-off between the number of subnets and the model's performance, as well as explore details of the training dynamics that allow the training of few subnets to be competitive with standard dropout.

## Repo Structure

```bash
├── k_dropout
│   ├── datasets.py
│   ├── experiment_helpers.py
│   ├── modules.py
│   ├── networks.py
│   └── training_helpers.py
├── notebooks
├── scripts
├── sweep_configs
├── README.md
├── ensemble_subnet_analysis.py
├── pooled_subnet_analysis.py
├── sequential_subnet_experiment.py
├── standard_subnet_experiment.py
├── test_modules.py
├── train_net.py
└── wandb_helpers.py
```

### `k_dropout`

The `k_dropout` directory contains the python code for creating and running models with k-dropout layers.

- `k_dropout/datasets.py` contains code for loading and preprocessing the MNIST and CIFAR-10 datasets.
- `k_dropout/experiment_helpers.py` contains functions for creating pytorch datasets and k-dropout modules with specific hyperparameters. These are called in the experiment files at the project root.
- `k_dropout/modules.py` contains the implementation of the sequential and pooled k-dropout pytorch modules. 
- `k_dropout/networks.py` contains functions for creating various pytorch models using our custom layers including a general `make_net()` and the `PoolDropoutLensNet` which can be used to control and analyze the subnetworks in a pooled k-dropout model.
- `k_dropout/training_helpers.py` contains functions for training and testing a model on dataset using specific hyperparameters and logging the data to weights and biases.

### `notebooks`

The `notebooks` directory contains the notebooks for plotting results. `notebooks/paper_plots.ipynb` specifically contains the final plots for the write-up.

### `scripts` and `sweep_configs`

The `scripts` and `sweep_configs` directories contain bash scripts and yaml files with the configuration for running experiments using weights and biases. 

### Project Root

The project root contains various files for running experiments.

- `ensemble_subnet_analysis.py` is the target for the pooled k-dropout ensemble experiment. 
- `pooled_subnet_analysis.py` is the target for the experiment tracking the performance of subnets in a pooled k-dropout model.
- `sequential_subnet_experiment.py` is the target for the experiment tracking the pserformance of subnets in a sequential k-dropout model.
- `standard_subnet_experiment.py` is the target for the experiment training the performance of subnets in a standard or no dropout model.
- `test_modules.py` contains functions for verifying the correct implementation of the sequential and pooled modules.
- `train_net.py` is the target for experiments that involve training a model with a set dropout module and hyperparameters and is used for the parameter sweeps.
- `wandb_helpers.py` contains functions that store additional information in the runs on weights and biases.

