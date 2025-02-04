import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import sys
import math
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta
from collections import defaultdict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# creat custom rnn

class CustomRNN(nn.Module):
    def __init__(self, input_sz, neuron_sz, synapse_sz, glia_sz, gama, tau, out_sz=3, seed=None, *args, **kwargs):
        super().__init__()
        self.input_size = input_sz
        self.neuron_size = neuron_sz
        self.synapse_size = synapse_sz
        self.glia_size = glia_sz
        if seed is not None:
            # use the provided seed value
            self.seed = seed
        else:
            # use a random seed
            self.seed = random.randint(0, 100)
        self.gama = gama
        self.tau = tau
        self.W_in1 = nn.Parameter(torch.Tensor(input_sz, neuron_sz))  # input matrix 1
        self.W_in2 = nn.Parameter(torch.Tensor(input_sz, glia_sz), requires_grad=True)
        self.C = nn.Parameter(torch.Tensor(synapse_sz, ))  # neuron to synapse connection matrix
        self.D = nn.Parameter(torch.Tensor(glia_sz, synapse_sz))  # glia to synapse connection matrix
        self.F = nn.Parameter(torch.Tensor(glia_sz, glia_sz))  # glia connection matrix
        self.H = nn.Parameter(torch.Tensor(synapse_sz, glia_sz))  # neuron to glia connection matrix

        # Initializing the parameters to some random values using normal distribution
        with torch.no_grad():  # this is to say that initialization will not be considered when computing the gradient later on
            self.W_in1.normal_()
            self.W_in2.normal_()
            self.C.normal_(std=1 / np.sqrt(self.synapse_size))
            self.D.normal_(std=1 / np.sqrt(self.synapse_size))
            self.F.normal_(std=1 / np.sqrt(self.glia_size))
            self.H.normal_(std=1 / np.sqrt(self.glia_size))


    def forward(self, inputs, init_states=None):

        if init_states is None:
            x_t, w_t, z_t = (torch.zeros(1, self.neuron_size).to(device),
                             torch.zeros(1, self.synapse_size).to(device),
                             torch.zeros(1, self.glia_size).to(device))
        else:
            x_t, w_t, z_t = init_states

        x_t = (1 - self.gama) * x_t + self.gama * (
                    torch.sigmoid(x_t) @ torch.reshape(w_t, (self.neuron_size, self.neuron_size)) + inputs @ self.W_in1)
        w_t = (1 - self.gama) * w_t + self.gama * (
                    torch.reshape((torch.sigmoid(x_t).reshape(-1, 1) @ torch.sigmoid(x_t)), (1, -1)) @ torch.diag(
                self.C) + torch.tanh(z_t) @ self.D)
        z_t = (1 - self.gama * self.tau) * z_t + self.gama * self.tau * (
                    torch.tanh(z_t) @ self.F + torch.reshape((torch.sigmoid(x_t).reshape(-1, 1) @ torch.sigmoid(x_t)),
                                                             (1, -1)) @ self.H + inputs @ self.W_in2 )

        new_hidden = (x_t, w_t, z_t)

        return x_t, new_hidden


class MyNet(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.netmodel = CustomRNN(input_sz=1, neuron_sz=128, synapse_sz=128 * 128, glia_sz=64, gama=0.2, tau=0.02,
                                  output_sz=3, seed=None).cuda()
        if seed is not None:
            # use the provided seed value
            self.seed = seed
        else:
            # use a random seed
            self.seed = random.randint(0, 100)
        self.fc = nn.Linear(128, 3)

    def forward(self, h, hidden=None):
        out, new_hidden = self.netmodel(h, hidden)
        out = self.fc(out)
        return out, new_hidden


# neuroglia model and training


class NeuroAstroRL(object):
    def __init__(self, input_sz=1, output_sz=3, seed=None):
        super().__init__()
        if seed is not None:
            # use the provided seed value
            self.seed = seed
        else:
            # use a random seed
            self.seed = random.randint(0, 100)
        self.model = MyNet(seed=self.seed).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.n_actions = output_sz
        self.state_size = output_sz

        self.reset()

    def reset(self):
        self.N = np.zeros((self.n_actions))
        self.action_bank = (torch.ones(self.n_actions, dtype=torch.float) * 10).cuda()

    # training
    def train(self, ds):
        self.model.train()
        rewards = [0]
        regrets = []
        reward_dict = defaultdict(float)
        init_states = None
        new_hidden = None

        loss = 0
        mean_regret = 0
        cumulative_regret = 0
        cumulative_regrets = []

        for i, (rew, orig_probs) in enumerate(ds):
            action, reward, regret, new_hidden, m, logits = self.get_arm(rew, orig_probs, ds,
                                                                         ht=new_hidden)
            m_reward = (reward - reward_dict[0])

            loss += - m_reward * m.log_prob(action.cuda())  # loss function L_R=-(r_it,t-bar_r_t)log p_it.

            loss.backward()
            self.optimizer.step()
            if isinstance(new_hidden, tuple):
                new_hidden = tuple(nh.detach() for nh in new_hidden)
            else:
                new_hidden = new_hidden.detach()

            loss = 0
            self.optimizer.zero_grad()

            self.N[action] += 1
            regrets.append(float(regret))
            rewards.append(rewards[- 1] + float(reward))

            cumulative_regret += regret
            cumulative_regrets.append(cumulative_regret)  # cumulative regret

            n = self.N.sum().item()
            reward_dict[0] = (reward_dict[0] * (n - 1) + reward) / (n)
            mean_regret = (mean_regret * (n - 1) + regret) / (n)

        return rewards, regrets, cumulative_regrets

    # choose actions

    def get_arm(self, rew, orig_probs, ds, **kwargs):
        old_state = kwargs['ht']
        # context = kwargs['context']

        dummy_state = (torch.ones(1)).unsqueeze(0).cuda()
        env_cue = dummy_state  # this is only for the stationary case, for non-stationary cases, the cue should be adjusted accordingly {-1,0,1}
        logits, new_hidden = self.model(env_cue, old_state)

        if self.model.training:
            m = Categorical(logits=logits / 2.0)
            action = m.sample()
        else:
            _, action = torch.max(logits, 2)


        action = action.detach().cpu()
        reward = rew[0, action].detach().cpu().item() if self.action_bank.detach().cpu()[action] > 0 else 0

        regret = np.max(orig_probs.squeeze().numpy() * np.clip(self.action_bank.detach().cpu().numpy(), 0, 1)) - \
                 (orig_probs.squeeze().numpy() * np.clip(self.action_bank.detach().cpu().numpy(), 0, 1))[action]

        return action, reward, regret, new_hidden, m, logits
