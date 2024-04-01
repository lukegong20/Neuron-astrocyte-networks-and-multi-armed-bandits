import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributions.bernoulli import Bernoulli
import numpy as np

torch.manual_seed(0)

# stationary bandit
class StationaryBbs(Dataset):
    def __init__(self, , n_samples=10001, n_actions=3, lamada=0.05):
        super().__init__()
        self.n_samples = n_samples
        self.dist_parameters = torch.Tensor([0.4, 0.8, 0.1])
#         self.dist_parameters = torch.Tensor([0.6-lamada, 0.6, 0.6+lamada]) # for robust analysis
        self.randomizer = Bernoulli(probs=self.dist_parameters)
        self.n_actions = n_actions


    def __getitem__(self, item):
        return self.randomizer.sample(), self.dist_parameters

    def __len__(self):
        return self.n_samples
    


    
# non-stationary bandit--abrupt change
class Flip_flop(Dataset):
    def __init__(self, n_samples=10001, n_actions=3, period=2000):
        super().__init__()
        self.n_samples = n_samples
        self.dist_parameters = torch.rand((n_actions))
        self.phase = 2 * np.pi / n_actions
        self.omega = 2 * np.pi / period
        self.n_actions = n_actions
        

    def __getitem__(self, item):
        probs = torch.tensor([self.dist_parameters[i] * (np.sign(np.sin(self.omega * item + self.phase * i))+1.1)/2.1 for i in
                              range(self.n_actions)]) 
        self.randomizer = Bernoulli(probs=probs)
        return self.randomizer.sample(), probs

    def __len__(self):
        return self.n_samples

# non-stationary bandit-smooth change
class Smooth_change(Dataset):
    def __init__(self, n_samples=80001, n_actions=3, period=5000):
        super().__init__()
        self.n_samples = n_samples
        self.dist_parameters = torch.rand((n_actions))
        self.phase = 2 * np.pi / n_actions
        self.omega = 2 * np.pi / period
        self.n_actions = n_actions


    def __getitem__(self, item):
        probs = torch.tensor([self.dist_parameters[i]  / (1 + np.exp(-50 * np.sin(self.omega * item + self.phase * i)))  for i in
                              range(self.n_actions)])   # sigmoild function periodic non-stationary
        self.randomizer = Bernoulli(probs=probs)
        return self.randomizer.sample(), probs

    def __len__(self):
        return self.n_samples
