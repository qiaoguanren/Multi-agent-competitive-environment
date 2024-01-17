import torch
import math, random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from tqdm import tqdm
from utils import VAE


def layer_init(layer, std=np.sqrt(2), bias_const=1.0):
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
    
    def forward(self, state, scale):

        mean = self.f(state) + scale
        var = F.softplus(self.f(state)) + 1e-8
        dist = Normal(mean, var)
        normal_sample = dist.rsample()
        # log_prob = dist.log_prob(normal_sample)
        # log_prob = log_prob - torch.log(1 - torch.tanh(normal_sample).pow(2) + 1e-7)
        # log_prob = log_prob.sum(1, keepdim=True)
        logp_pi = dist.log_prob(normal_sample).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - normal_sample - F.softplus(-2*normal_sample))).sum(axis=1)
        log_prob = logp_pi
        
        return normal_sample, log_prob
    
class QValueNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QValueNet, self).__init__()
        self.fc = nn.Sequential(
            layer_init(nn.Linear(state_dim+action_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(hidden_dim, hidden_dim//2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(hidden_dim//2, 1)),

        )# two-layer MLP

    def forward(self, states):

        v = self.fc(states)
        return v
    
class SAC:
    def __init__(self, 
                 state_dim: int, 
                 agent_index: int,
                 hidden_dim: int,
                 action_dim: int,
                 batchsize: int,
                 actor_learning_rate: float,
                 critic_learning_rate: float,
                 alpha_learning_rate: float,
                 density_learning_rate: float,
                 gamma: float, 
                 device, 
                 agent_num: int,
                 offset: int, 
                 tau: float,
                 beta_coef: float,
                 kld_weight: float,
                 epochs: int):
        self.actor = Policy(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1 = QValueNet(state_dim, action_dim, hidden_dim).to(device)
        self.critic_2 = QValueNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim,action_dim,hidden_dim).to(device)
        self.target_critic_2 = QValueNet(state_dim,action_dim,hidden_dim,).to(device)
        self.vae = VAE(state_dim, hidden_dim//4).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.batchsize = batchsize
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.gamma = gamma
        self.device = device
        self.agent_num = agent_num
        self.agent_index = agent_index
        self.offset = offset
        self.epochs = epochs
        self.tau = tau
        self.beta_coef = beta_coef
        self.kld_weight = kld_weight

        self.density_optimizer = torch.optim.Adam(self.vae.parameters(), lr=density_learning_rate)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(),
                                                lr=actor_learning_rate)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_learning_rate)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_learning_rate)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_learning_rate)
        self.target_entropy = -action_dim
        
    def sample(self, transition, sample_batchsize, buffer_batchsize, offset):

        s = []
        a = []
        r = []
        s_next = []
        done = []

        with torch.no_grad():
            for _ in range(sample_batchsize):

                row = random.randint(0,buffer_batchsize-1)
                col = random.randint(0,int(60/offset)-1)
                s.append(transition[row]['states'][col])
                a.append(transition[row]['actions'][col])
                r.append(transition[row]['rewards'][col])
                s_next.append(transition[row]['next_states'][col])
                done.append(transition[row]['dones'][col])

        return s, a, r, s_next, done

    def choose_action(self, state, scale):
        with torch.no_grad():
            state = torch.flatten(state,start_dim=0).unsqueeze(0)
            action, _ = self.actor(state, scale)
        return action
    
    def get_target(self, rewards, next_states, dones, beta_coef, kld_weight, scale):
        with torch.no_grad():
            actions, log_prob = self.actor(next_states, scale)
            entropy = -log_prob.unsqueeze(-1)
            next_Q_input = torch.cat([next_states, actions], dim=-1)
            q1_value = self.target_critic_1(next_Q_input)
            q2_value = self.target_critic_2(next_Q_input)
            next_value = torch.min(q1_value,
                                  q2_value) + self.log_alpha.exp() * entropy
            td_target = rewards + self.gamma * next_value * (1 - dones)

        density, output_t, mu, logvar = self.vae(next_states)

        density_loss = self.vae.loss_function(next_states, output_t, mu, logvar, kld_weight)
        self.density_optimizer.zero_grad()
        density_loss.mean().backward()
        self.density_optimizer.step()

        beta_t = beta_coef / density
        td_target = td_target + beta_t
        return td_target
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)
    
    def update(self, transition, buffer_batchsize, scale):
        
        for epoch in tqdm(range(self.epochs)):

            states,  actions, rewards, next_states, dones= self.sample(transition, self.batchsize, buffer_batchsize, self.offset)
            states = torch.stack(states, dim=0).flatten(start_dim=1)
            next_states = torch.stack(next_states, dim=0).flatten(start_dim=1)
            rewards = torch.stack(rewards, dim=0).view(-1,1)
            dones = torch.stack(dones, dim=0).view(-1,1)
            actions = torch.stack(actions, dim=0).flatten(start_dim=1)

            Q_input = torch.cat([states, actions], dim=-1)
            td_target = self.get_target(rewards, next_states, dones, self.beta_coef, self.kld_weight, scale)
            critic_1_loss = torch.mean(
                F.mse_loss(self.critic_1(Q_input), td_target.detach()))
            critic_2_loss = torch.mean(
                F.mse_loss(self.critic_2(Q_input), td_target.detach()))
            self.critic_1_optimizer.zero_grad()
            critic_1_loss.backward()
            self.critic_1_optimizer.step()
            self.critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            self.critic_2_optimizer.step()

            new_actions, log_prob = self.actor(states, scale)
            entropy = -log_prob
            new_Q_input = torch.cat([states, new_actions], dim=-1)
            q1_value = self.critic_1(new_Q_input)
            q2_value = self.critic_2(new_Q_input)
            actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                    torch.min(q1_value, q2_value))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            self.soft_update(self.critic_1, self.target_critic_1)
            self.soft_update(self.critic_2, self.target_critic_2)

        
