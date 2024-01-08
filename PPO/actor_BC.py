import torch
import math, random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from tqdm import tqdm


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

class Policy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Policy, self).__init__()
        self.f = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(hidden_dim, action_dim)),
        )
    
    def forward(self, state):

        mean = self.f(state)
        var = F.sigmoid(self.f(state)) + 1e-8
        return mean, var
    
class BC:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, epochs, batchsize, offset):
        self.action_dim = action_dim
        self.lr = lr
        self.batchsize = batchsize
        self.offset = offset
        self.epochs = epochs
        self.actor = Policy(state_dim, hidden_dim, action_dim).cuda()
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam([
                        {'params': self.actor.parameters(), 'lr': self.lr}])

    def sample(self, transition, sample_batchsize, buffer_batchsize, offset):

        s = []
        a = []

        with torch.no_grad():

            for _ in range(sample_batchsize):

                row = random.randint(0,buffer_batchsize-1)
                col = random.randint(0,int(60/offset)-1)
                s.append(transition[row]['states'][col])
                a.append(transition[row]['actions'][col])

        return s, a

    def train(self, transition, buffer_batchsize):
        for epoch in tqdm(range(self.epochs)):
            states,  actions = self.sample(transition, self.batchsize, buffer_batchsize, self.offset)
            states = torch.stack(states, dim=0).flatten(start_dim=1)
            actions = torch.stack(actions, dim=0).flatten(start_dim=1)
            
            mean, var = self.actor(states)
            # action_distribution = Normal(mean, var)
            predicted_actions = mean
            loss = self.loss_function(predicted_actions, actions.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

