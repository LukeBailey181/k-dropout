import torch
from networks import PoolDropoutLensNet
from training_helpers import test_net
import wandb 

# TODO: implement plot_train_loss (using wandb and maybe without it too)
def plot_train_loss():
    pass

def evaluate_subnets_in_pooled_dropout_net(path_to_net, num_subnets, testset):

    # Init lens net using the loaded model
    net = torch.load(path_to_net)
    lens_net = PoolDropoutLensNet(init_net=net)

    # Get pooled subnet accs
    pooled_subnet_accs = []
    for subnet_idx in num_subnets: 
        lens_net.freeze_mask(subnet_idx)
        _, test_acc = test_net(lens_net, testset)
        pooled_subnet_accs.append(test_acc)

    # Do wandb logging for pooled subnets
    subnet_idxs = list(range(num_subnets))
    wandb_plot(
        "pooled_subnet_accs", 
        "Pooled Subnet Test Accuracy",
        "Subnet Index",
        "Test Accuracy", 
        subnet_idxs,
        pooled_subnet_accs,
        "scatter"
    )
    
    # Get random subnet accs
    lens_net.activate_random_masking()
    print("Switched net to random droput:")
    print(lens_net)

    random_subnet_accs = []
    for _ in num_subnets: 
        # Get accuracy when using dropout net by setting eval_net=False 
        _, test_acc = test_net(lens_net, testset, eval_net=False)
        random_subnet_accs.append(test_acc)

    # Do wandb logging for random subnets
    wandb_plot(
        "random_subnet_accs", 
        "Random Subnet Test Accuracy",
        "Subnet Index",
        "Test Accuracy", 
        subnet_idxs,
        random_subnet_accs,
        "scatter"
    )

def evaluate_subnets_in_dropout_net(path_to_net, num_subnets, testset):

    # Init lens net using the loaded model
    net = torch.load(path_to_net)

    pass


#-----wandb plotting helper functions-----#
def wandb_plot(plot_id, plot_title, x_label, y_label, x_vals, y_vals, plot_style):
    """Helper function for custom wandb plots
    
    Arguments:
        plot_id: string id of plot for wandb
        plot_title: string plot title
        x_label: x axis label
        y_label: y axis label 
        x_vals: list of x values
        y_vals: list of y values
        plot_style: one of {line, scatter} indicating style of plot.
    """

    if plot_style == "line":
        plot_func = wandb.plot.line
    elif plot_style == "scatter":
        plot_func = wandb.plot.scatter
    else:
        raise RuntimeError(f"plot_style of {plot_style} not in ('line', 'scatter')")

    data = [[x, y] for (x, y) in zip(x_vals, y_vals)]
    table = wandb.Table(data=data, columns = [x_label, y_label])
    wandb.log( {plot_id : plot_func(table, x_label, y_label, title=plot_title)})

if __name__ == "__main__": 
    # TODO do some arg parsing here
    pass
