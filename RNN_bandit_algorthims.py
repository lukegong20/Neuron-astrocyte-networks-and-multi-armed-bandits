import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import sys
import math
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta





""""vanilla_RNN""""
class RNNnet(nn.Module):
    def __init__(self, in_size, out_size, seed=None, *args, **kwargs):
        super().__init__()
        if seed is not None:
            # use the provided seed value
            self.seed = seed
        else:
            # use a random seed
            self.seed = random.randint(0, 100)
        self.seq = nn.RNN(in_size, hidden_size=128, num_layers=2,
                                      batch_first=True)
        self.fc = nn.Linear(128, out_size)
        
        

    def forward(self, x, hidden=None):
        
        out, new_hidden = self.seq(x, hidden)
        
        return self.fc(out), new_hidden


""""GRU_RNN""""
class GRUnet(nn.Module):
    def __init__(self, in_size, out_size, seed=None, *args, **kwargs):
        super().__init__()
        if seed is not None:
            # use the provided seed value
            self.seed = seed
        else:
            # use a random seed
            self.seed = random.randint(0, 100)
        self.seq = nn.GRU(in_size, hidden_size=128, num_layers=2,
                                      batch_first=True)
        self.fc = nn.Linear(128, out_size)
        
        

    def forward(self, x, hidden=None):
        
        out, new_hidden = self.seq(x, hidden)
        
        return self.fc(out), new_hidden



"""LSTM_RNN"""
class LSTMnet(nn.Module):
    def __init__(self, in_size, out_size, seed=None, *args, **kwargs):
        super().__init__()
        if seed is not None:
            # use the provided seed value
            self.seed = seed
        else:
            # use a random seed
            self.seed = random.randint(0, 100)
        self.seq = nn.LSTM(in_size, hidden_size=128, num_layers=2,
                                      batch_first=True)
        self.fc = nn.Linear(128, out_size)
        
        

    def forward(self, x, hidden=None):
        
        out, new_hidden = self.seq(x, hidden)
        
        return self.fc(out), new_hidden


