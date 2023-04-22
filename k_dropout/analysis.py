import matplotlib.pyplot as plt
import torch
from networks import PoolDropoutLensNet

# TODO: implement plot_train_loss (using wandb and maybe without it too)
def plot_train_loss():
    pass

def evaluate_subnets_in_pooled_dropout_net(path_to_net, num_subnets):

    # Init lens net using the loaded model
    net = torch.load(path_to_net)
    lens_net = PoolDropoutLensNet(init_net=net)

    for subnet_idx in num_subnets: 
        lens_net.freeze_mask(subnet_idx)
        # Do testing
        # Do WandB logging
    
    for _ in num_subnets: 
        # Create a random subnet
        # Test this
        pass

def evaluate_subnets_in_dropout_net():
    pass

if __name__ == "__main__": 
    # TODO do some arg parsing here
    pass
