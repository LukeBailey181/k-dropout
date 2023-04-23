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

    pooled_subnet_accs = []
    for subnet_idx in num_subnets: 
        lens_net.freeze_mask(subnet_idx)
        _, test_acc = test_net(lens_net, testset)
        pooled_subnet_accs.append(test_acc)

    # Do wandb logging for pooled subnets
    subnet_idxs = list(range(num_subnets))
    data = [[x, y] for (x, y) in zip(subnet_idxs, pooled_subnet_accs)]
    table = wandb.Table(data=data, columns = ["x", "y"])
    wandb.log(
        {
            "pooled_subnet_accs" : wandb.plot.scatter(
                table, 
                "Subnet index", 
                "Test Accuracy", 
                title="Pooled Subnet Test Accuracies"
            )
        }
    )
    
    for _ in num_subnets: 
        # Create a random subnet
        # Test this
        pass

def evaluate_subnets_in_dropout_net():
    pass


#-----wandb plotting helper functions-----#

def wandb_plot(plot_id, plot_title, x_label, y_lable, x_vals, y_vals):
    pass

if __name__ == "__main__": 
    # TODO do some arg parsing here
    pass
