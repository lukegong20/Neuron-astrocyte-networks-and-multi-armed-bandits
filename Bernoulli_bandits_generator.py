import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributions.bernoulli import Bernoulli
import numpy as np

torch.manual_seed(0)

class TimeIndependent(Dataset):
    def __init__(self, n_contexts=1, n_samples=60000, n_actions=10):
        super().__init__()
        self.n_samples = n_samples
        self.dist_parameters = torch.rand((n_actions))  # random probabilities
#         self.dist_parameters = torch.Tensor([0.7, 0.5, 0.3])
        self.randomizer = Bernoulli(probs=self.dist_parameters)
        self.n_actions = n_actions
        self.n_contexts = n_contexts

    def __getitem__(self, item):
        return self.randomizer.sample(), self.dist_parameters, [0], [0]

    def __len__(self):
        return self.n_samples


class TimeDependentSinus(Dataset):
    def __init__(self, n_contexts=1, n_samples=60000, n_actions=10, period=10000):
        super().__init__()
        self.n_samples = n_samples
        self.dist_parameters = torch.rand((n_actions))
        self.phase = 2 * np.pi / n_actions
        self.omega = 2 * np.pi / period
        self.n_actions = n_actions
        self.n_contexts = n_contexts

    def __getitem__(self, item):
#         probs = torch.tensor([self.dist_parameters[i] * np.abs(np.sin(self.omega * item + self.phase * i)) for i in
#                               range(self.n_actions)])         # sin function
        probs = torch.tensor([self.dist_parameters[i]  / (1 + np.exp(-100 * np.sin(self.omega * item + self.phase * i)))  for i in
                              range(self.n_actions)])   # sigmoild function periodic non-stationary
        self.randomizer = Bernoulli(probs=probs)
        return self.randomizer.sample(), probs, [0], [0]

    def __len__(self):
        return self.n_samples
