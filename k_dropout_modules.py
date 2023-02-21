import torch
from torch import nn
import torch.nn.functional as F
from helpers import train_net, test_net, get_mnist, StandardNet

class StochasticKDropout(nn.Linear):
    def __init__(self, in_feats, out_feats, drop_p, k, bias=True):
        super(StochasticKDropout, self).__init__(in_feats, out_feats, bias=bias)

        self.p = drop_p
        self.k = k
        self.uses = 0
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
        self.mask = None

    def forward(self, input):
        
        Z = F.linear(input, self.weight, self.bias)

        if self.training: 

            if self.uses % self.k == 0:
                # Update mask 
                self.mask = self.binomial.sample(Z.size()).to(self.weight.device)
                self.uses = 0

            self.uses += 1

            # Dropout activations and scale the rest
            return Z * self.mask * (1.0/(1-self.p))

        return Z 


class RRKDropout(nn.Linear):
    def __init__(self, in_feats, out_feats, drop_p, k, num_masks, bias=True):
        super(RRKDropout, self).__init__(in_feats, out_feats, bias=bias)

        self.p = drop_p
        self.k = k
        self.uses = 0
        self.mask_seeds = torch.randint(high=5000, size=(num_masks,))
        self.mask_idx = -1
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
        self.mask = None

    def forward(self, input):
        
        Z = F.linear(input, self.weight, self.bias)

        if self.training: 

            if self.uses % self.k == 0:
                # Update index
                self.mask_idx += 1
                self.mask_idx %= self.mask_seeds.shape[0]

                # Get same mask as previous mask_idx by seeding RNG
                torch.manual_seed(self.mask_seeds[self.mask_idx].item())
                self.mask = self.binomial.sample(Z.size()).to(self.weight.device)
                self.uses = 0

            self.uses += 1

            # Dropout activations and scale the rest
            return Z * self.mask * (1.0/(1-self.p))

        return Z 


class StochasticKDropoutNet(nn.Module):

    def __init__(self, num_classes=2, input_dim=2, hidden_units=100, drop_p=0.5, k=1):
        super(StochasticKDropoutNet, self).__init__()
        self.fc1 = StochasticKDropout(input_dim, hidden_units, drop_p, k)  
        self.fc2 = StochasticKDropout(hidden_units, hidden_units, drop_p, k)  
        self.fc3 = StochasticKDropout(hidden_units, hidden_units, drop_p, k)  
        self.fc4 = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


if __name__ == "__main__":

    # Train standard net
    train_loader, test_loader = get_mnist()
    standard_net = StandardNet(num_classes=10, input_dim=784)
    train_net(20, standard_net, train_loader)
    
    # Train kdropout net
    train_loader, test_loader = get_mnist()
    dropout_net = StochasticKDropoutNet(num_classes=10, input_dim=784, drop_p=0, k=10**10)
    train_net(10, dropout_net, train_loader)

    _, standard_acc = test_net(standard_net, test_loader)
    _, dropout_acc = test_net(dropout_net, test_loader)
    print(f"Standard Net MNIST testing accuracy: {standard_acc}")
    print(f"Dropout Net MNIST testing accuracy: {dropout_acc}")
