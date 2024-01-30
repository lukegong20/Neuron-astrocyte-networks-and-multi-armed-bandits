import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import sys
import math

import os.path
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta
from tqdm import tqdm, trange
from collections import defaultdict
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import Dataset, DataLoader, Subset

class thompson_sampling():





class UCB():




class SW_UCB():



class D_UCB():
