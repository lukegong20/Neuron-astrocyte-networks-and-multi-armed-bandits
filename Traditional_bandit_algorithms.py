import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import sys
import math
import pandas as pd
import seaborn as sns
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta
from collections import defaultdict
from torch.distributions.bernoulli import Bernoulli


"""Standard Thompson Sampling"""
class TS(object):
    def __init__(self, n_actions=3, seed=None, *args, **kwargs):
        super().__init__()
        if seed is not None:
            # use the provided seed value
            self.seed = seed
        else:
            # use a random seed
            self.seed = random.randint(0, 100)
        self.n_actions = n_actions
        self.alphas = np.ones(n_actions)
        self.betas = np.ones(n_actions) + sys.float_info.epsilon
        self.n_actions = n_actions
        
        self.reset()

    def reset(self):
        self.t = 0
        self.N = np.zeros(self.n_actions, dtype=np.int32)
        self.action_bank = (torch.ones(self.n_actions, dtype=torch.float) * 10).cuda()

    def train(self, ds):
        rewards = [0]
        regrets = []
        mean_regret = 0
        cumulative_regret = 0
        cumulative_regrets =[]

        for i, (rew, orig_probs, mask) in enumerate(ds):
            action, reward, regret, _, _ = self.get_arm(rew, orig_probs, mask, ds)
            self.t += 1
            self.alphas[action] += reward
            self.betas[action] += 1 - reward
            self.N[action] += 1

            regrets.append(float(regret))
            rewards.append(rewards[- 1] + float(reward))

            cumulative_regret += regret
            cumulative_regrets.append(cumulative_regret)

            mean_regret = (mean_regret * i + regret) / (i + 1)

        return rewards, regrets, cumulative_regrets

    def theta(self):
        return self.alphas / (self.alphas + self.betas)

    def get_arm(self, rew, orig_probs, mask, ds, **kwargs):
        if self.t < self.n_actions:
            #for t from 1 to n_action, choose the arm a_t=t;  
            #for t from n+1 to T, choose an arm according to argmax{discounted_reward+bonus}
            action = self.t
        else:            
            
            beta_dist = Beta(torch.tensor(self.alphas), torch.tensor(self.betas))
            theta = beta_dist.sample()
            action = np.argmax(theta.numpy())

        reward = rew[0, action].item() if self.action_bank[action] > 0 else 0

        regret = np.max(orig_probs.squeeze().numpy() * np.clip(self.action_bank.cpu().numpy(), 0, 1)) - \
                 (orig_probs.squeeze().numpy() * np.clip(self.action_bank.cpu().numpy(), 0, 1))[action]

        
        return action, reward, regret, None, None






"""standard UCB for stationary Bbs"""
class UCB(object):
    def __init__(self, n_actions=3,seed=None, *args, **kwargs):
        super().__init__()
        if seed is not None:
            # use the provided seed value
            self.seed = seed
        else:
            # use a random seed
            self.seed = random.randint(0, 100)
        self.n_actions = n_actions
        self.successcount = np.zeros(n_actions)
        self.failcount = np.zeros(n_actions) 
        self.n_actions = n_actions
        
        self.reset()

    def reset(self):
        self.t = 0
        self.N = np.zeros(self.n_actions, dtype=np.int32)
        self.action_bank = (torch.ones(self.n_actions, dtype=torch.float) * 10).cuda()

    def train(self, ds):
        rewards = [0]
        regrets = []
        mean_regret = 0
        cumulative_regret = 0
        cumulative_regrets =[]
        for i, (rew, orig_probs, mask) in enumerate(ds):
            action, reward, regret, _, _ = self.get_arm(rew, orig_probs, mask, ds)
            self.t += 1
            self.successcount[action] += reward
            self.failcount[action] += 1 - reward
            self.N[action] += 1

            regrets.append(float(regret))
            rewards.append(rewards[- 1] + float(reward))

            cumulative_regret += regret
            cumulative_regrets.append(cumulative_regret)
                
        return rewards, regrets, cumulative_regrets

   

    def get_arm(self, rew, orig_probs, mask, ds, **kwargs):
        if self.t < self.n_actions:
            #for t from 1 to n_action, choose the arm a_t=t;  
            #for t from n+1 to T, choose an arm according to argmax{discounted_reward+bonus}
            action = self.t
        else:  
            success_ratio = self.successcount / (self.successcount + self.failcount)    
            # computing square root term
            sqrt_term = np.sqrt(2*np.log(np.sum(self.N))/(self.successcount + self.failcount))
            action = np.argmax(success_ratio + sqrt_term)

        reward = rew[0, action].item() if self.action_bank[action] > 0 else 0
        regret = np.max(orig_probs.squeeze().numpy() * np.clip(self.action_bank.cpu().numpy(), 0, 1)) - \
                 (orig_probs.squeeze().numpy() * np.clip(self.action_bank.cpu().numpy(), 0, 1))[action]

        return action, reward, regret, None, None


"""Two extentions of UCB for non-stationary Bbs: Discounted / Sliding-window UCBs from work
    'On Upper-Confidence Bound Policies for Switching Bandit Problems (2011)' """

"""Discounted UCB"""

class DUCB(object):
    def __init__(self,  n_actions=3, gamma=0.95, ksi=0.6, *args, **kwargs):
        super().__init__()
        self.n_actions = n_actions
        self.gamma = gamma
        self.ksi = ksi
        self.reset()

    def reset(self):
        self.history = [] # history of played arms, of size t
        self.history_bool = None #  history of played arms as booleans, of size (t,n_actions)
        self.reward_history = []
        self.t = 0
        self.N = np.zeros(self.n_actions, dtype=np.int32)
        self.action_bank = (torch.ones(self.n_actions, dtype=torch.float) * 10).cuda()

    def train(self, ds):
        rewards = [0]
        regrets = []
        mean_regret = 0
        cumulative_regret = 0
        cumulative_regrets =[]           

        for i, (rew, orig_probs, mask) in enumerate(ds):

            action, reward, regret, _, _ = self.get_arm(rew, orig_probs, mask, ds)
            self.t += 1
            self.history.append(action)
            self.reward_history.append(reward)
            arm_bool = np.zeros(self.n_actions)
            arm_bool[action] = 1

            if self.history_bool is None:
                self.history_bool = arm_bool
            else : 
                self.history_bool = np.vstack((self.history_bool,arm_bool))

            self.N[action] += 1

            regrets.append(regret)
            rewards.append(rewards[-1] + reward)

            cumulative_regret += regret
            cumulative_regrets.append(cumulative_regret)

            mean_regret = (mean_regret * i + regret) / (i + 1)

        return rewards, regrets, cumulative_regrets

   

    def get_arm(self, rew, orig_probs, mask, ds, **kwargs):
        
        if self.t < self.n_actions:   
                action = self.t
        else:            

            discount = np.ones(self.t)*self.gamma**(self.t-np.arange(self.t))
            N = np.array([np.sum(discount[np.where(j==np.array(self.history),True,False)]) for j in range(self.n_actions)])
            #discounted average reward
            disc_reward = 1/N * np.sum(discount.reshape(-1,1) * np.reshape(self.reward_history, (-1,1)) * self.history_bool, axis=0) # discounted empirical average
            #discounted bonus
            bonus = 2  * np.sqrt((self.ksi * np.log(N.sum())/N))# discounted padding function   
            # choose action
            action = np.argmax(disc_reward + bonus)

        reward = rew[0, action].item() if self.action_bank[action] > 0 else 0
        regret = np.max(orig_probs.squeeze().numpy() * np.clip(self.action_bank.cpu().numpy(), 0, 1)) - \
                 (orig_probs.squeeze().numpy() * np.clip(self.action_bank.cpu().numpy(), 0, 1))[action]

        return action, reward, regret, None, None
    


""" Sliding-window UCB"""   
    
class SWUCB(object):
    def __init__(self, n_actions=3, tau=10, ksi=0.6, *args, **kwargs): 
        super().__init__()
        self.n_actions = n_actions
        self.tau = tau
        self.ksi = ksi
        
        self.reset()

    def reset(self):
        self.N = np.zeros(self.n_actions, dtype=np.int32)
        self.action_bank = (torch.ones(self.n_actions, dtype=torch.float) * 10).cuda()
        # history of played arms, of size t
        self.history = []
        
        # history of played arms as booleans, of size (t,n_actions)
        self.history_bool = None 
        self.t = 0
        # successive rewards, of size (t, n_actions), to keep track of them
        # in order to compute the sum X_t(tau, i), denoted by X here
        self.rewards1 = None
        
    def train(self, ds):
        rewards = [0]
        regrets = []
        mean_regret = 0
        cumulative_regret = 0
        cumulative_regrets =[]
        for i, (rew, orig_probs, mask) in enumerate(ds):

            action, reward, regret, _, _ = self.get_arm(rew, orig_probs, mask, ds)
            self.t += 1
            # add to history
            self.history.append(action)

            # add to history_bool
            arm_bool = np.zeros(self.n_actions)
            arm_bool[action] = 1
            if self.history_bool is None : 
                # this is for t=0
                # it is only a trick for initialization and then vstack
                self.history_bool = arm_bool
            else : 
                self.history_bool = np.vstack((self.history_bool,arm_bool))

            # add reward to self.rewards
            # same trick for the rewards
            reward_this_step = np.zeros(self.n_actions)
            reward_this_step[action] = reward
            if self.rewards1 is None:
                # first step, t=0
                self.rewards1 = reward_this_step
            else:
                self.rewards1 = np.vstack((self.rewards1,reward_this_step))


            self.N[action] += 1
            regrets.append(regret)
            rewards.append(rewards[-1] + reward)

            cumulative_regret += regret
            cumulative_regrets.append(cumulative_regret)

            mean_regret = (mean_regret * i + regret) / (i + 1)
        return rewards, regrets, cumulative_regrets

   

    def get_arm(self, rew, orig_probs, mask, ds, **kwargs):
        if self.t < self.n_actions:            
            action = self.t
        else:     
            N = np.sum(self.history_bool[-self.tau:], axis=0)
            #average reward
            disc_reward = (1/N) * np.sum(self.rewards1[-self.tau:], axis=0)
            #bonus
            bonus = np.sqrt((self.ksi * np.log(max(self.t, self.tau)))/N)
            bound = np.nan_to_num(disc_reward + bonus, copy=False, nan=np.inf)  # dealing with nan values
            # choose action
            action = np.argmax(bound)

        reward = rew[0, action].item() if self.action_bank[action] > 0 else 0
        regret = np.max(orig_probs.squeeze().numpy() * np.clip(self.action_bank.cpu().numpy(), 0, 1)) - \
                 (orig_probs.squeeze().numpy() * np.clip(self.action_bank.cpu().numpy(), 0, 1))[action]

        return action, reward, regret, None, None 

