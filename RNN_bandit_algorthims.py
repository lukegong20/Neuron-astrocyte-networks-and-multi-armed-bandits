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
        
class RecurrentRNN(object):
    def __init__(self, in_size=1, out_size=3, seed, *args, **kwargs):
        if seed is not None:
            # use the provided seed value
            self.seed = seed
        else:
            # use a random seed
            self.seed = random.randint(0, 100)
        super().__init__()
        self.model = RNNnet(in_size=1, out_size=3, seed=None, *args, **kwargs).cuda()
        self.state_size = out_size
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3) 
        
        self.n_actions = out_size  
        
        self.context_size = in_size
        self.reset()

    def reset(self):
        self.N = np.zeros((self.n_actions))
        self.action_bank = (torch.ones(self.n_actions, dtype=torch.float) * 10).cuda()

        
    def train(self, ds):
        self.model.train()
        rewards = [0]
        regrets = []
        reward_dict = defaultdict(float)
        init_states = None 
        new_hidden = None
        self.t = 0
        
        loss = 0
        mean_regret = 0
        cumulative_regret = 0
        cumulative_regrets = []
   
        for i, (rew, orig_probs, context, mask) in enumerate(ds):
            action, reward, regret, new_hidden, m, logits = self.get_arm(rew, orig_probs, mask, ds, context=context,
                                                                         ht=new_hidden)
            m_reward = (reward - reward_dict[0])
            self.t += 1
            loss += - m_reward * m.log_prob(action.cuda())   # loss function L_R=-(r_it,t-bar_r_t)log p_it # neural signals
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
            cumulative_regrets.append(cumulative_regret)  #cumulative regret

            n = self.N.sum().item()
            reward_dict[0] = (reward_dict[0] * (n - 1) + reward) / (n)
            mean_regret = (mean_regret * (n - 1) + regret) / (n)

        return rewards, regrets, cumulative_regrets

    
# choose actions

    def get_arm(self, rew, orig_probs, mask, ds, **kwargs):
        old_state = kwargs['ht']
        env_state = torch.exp(-torch.zeros(ds.dataset.n_contexts, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()) # feed inputs

        logits,  new_hidden = self.model(env_state, old_state)
    
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
class RecurrentGRU(object):
    def __init__(self, in_size=1, out_size=3, seed, *args, **kwargs):
        if seed is not None:
            # use the provided seed value
            self.seed = seed
        else:
            # use a random seed
            self.seed = random.randint(0, 100)
        super().__init__()
        self.model = GRUnet(in_size=1, out_size=3, seed=None, *args, **kwargs).cuda()
        self.state_size = out_size
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3) 
        
        self.n_actions = out_size  
        
        self.context_size = in_size
        self.reset()

    def reset(self):
        self.N = np.zeros((self.n_actions))
        self.action_bank = (torch.ones(self.n_actions, dtype=torch.float) * 10).cuda()

        
    def train(self, ds):
        self.model.train()
        rewards = [0]
        regrets = []
        reward_dict = defaultdict(float)
        init_states = None 
        new_hidden = None
        self.t = 0
        
        loss = 0
        mean_regret = 0
        cumulative_regret = 0
        cumulative_regrets = []
   
        for i, (rew, orig_probs, context, mask) in enumerate(ds):
            action, reward, regret, new_hidden, m, logits = self.get_arm(rew, orig_probs, mask, ds, context=context,
                                                                         ht=new_hidden)
            m_reward = (reward - reward_dict[0])
            self.t += 1
            loss += - m_reward * m.log_prob(action.cuda())   # loss function L_R=-(r_it,t-bar_r_t)log p_it # neural signals
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
            cumulative_regrets.append(cumulative_regret)  #cumulative regret

            n = self.N.sum().item()
            reward_dict[0] = (reward_dict[0] * (n - 1) + reward) / (n)
            mean_regret = (mean_regret * (n - 1) + regret) / (n)

        return rewards, regrets, cumulative_regrets

    
# choose actions

    def get_arm(self, rew, orig_probs, mask, ds, **kwargs):
        old_state = kwargs['ht']
        env_state = torch.exp(-torch.zeros(ds.dataset.n_contexts, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()) # feed inputs

        logits,  new_hidden = self.model(env_state, old_state)
    
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

class RecurrentLSTM(object):
    def __init__(self, in_size=1, out_size=3, seed, *args, **kwargs):
        if seed is not None:
            # use the provided seed value
            self.seed = seed
        else:
            # use a random seed
            self.seed = random.randint(0, 100)
        super().__init__()
        self.model = LSTMnet(in_size=1, out_size=3, seed=None, *args, **kwargs).cuda()
        self.state_size = out_size
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3) 
        
        self.n_actions = out_size  
        
        self.context_size = in_size
        self.reset()

    def reset(self):
        self.N = np.zeros((self.n_actions))
        self.action_bank = (torch.ones(self.n_actions, dtype=torch.float) * 10).cuda()

        
    def train(self, ds):
        self.model.train()
        rewards = [0]
        regrets = []
        reward_dict = defaultdict(float)
        init_states = None 
        new_hidden = None
        self.t = 0
        
        loss = 0
        mean_regret = 0
        cumulative_regret = 0
        cumulative_regrets = []
   
        for i, (rew, orig_probs, context, mask) in enumerate(ds):
            action, reward, regret, new_hidden, m, logits = self.get_arm(rew, orig_probs, mask, ds, context=context,
                                                                         ht=new_hidden)
            m_reward = (reward - reward_dict[0])
            self.t += 1
            loss += - m_reward * m.log_prob(action.cuda())   # loss function L_R=-(r_it,t-bar_r_t)log p_it # neural signals
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
            cumulative_regrets.append(cumulative_regret)  #cumulative regret

            n = self.N.sum().item()
            reward_dict[0] = (reward_dict[0] * (n - 1) + reward) / (n)
            mean_regret = (mean_regret * (n - 1) + regret) / (n)

        return rewards, regrets, cumulative_regrets

    
# choose actions

    def get_arm(self, rew, orig_probs, mask, ds, **kwargs):
        old_state = kwargs['ht']
        env_state = torch.exp(-torch.zeros(ds.dataset.n_contexts, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()) # feed inputs

        logits,  new_hidden = self.model(env_state, old_state)
    
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
