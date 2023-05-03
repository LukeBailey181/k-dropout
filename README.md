# CS 229 Final Project: K-Dropout

TODO: put paper abstract (or some introduction of the project) here

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

