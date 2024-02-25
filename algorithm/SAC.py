import torch
import math, random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Laplace
import numpy as np
from tqdm import tqdm


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
        mean = torch.cumsum(mean.reshape(-1,6,5,2), dim=-2)
        mean = mean[:, -1, :, :]
        mean = mean.flatten(start_dim = 1)

        b = self.f(state)
        b = torch.cumsum(F.elu_(b.reshape(-1,6,5,2),alpha = 1.0) + 1.0, dim=-2) + 0.1
        b = b[:, -1, :, :]
        b = b.flatten(start_dim = 1)

        # epsilon = 1e-6
        # mean = torch.where(torch.isnan(mean), torch.full_like(mean, epsilon), mean)
        # b = torch.where(torch.isnan(b), torch.full_like(b, epsilon), b)

        dist = Laplace(mean, b)

        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        log_prob = log_prob - torch.log(1 - torch.tanh(normal_sample).pow(2) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        # logp_pi = dist.log_prob(normal_sample).sum(axis=-1)
        # logp_pi -= (2*(np.log(2) - normal_sample - F.softplus(-2*normal_sample))).sum(axis=1)
        # log_prob = logp_pi
        
        return mean, b, normal_sample, log_prob
    
class QValueNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, agent_number):
        super(QValueNet, self).__init__()
        self.f = nn.Sequential(
            layer_init(nn.Linear(agent_number*state_dim+(action_dim//6), hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(hidden_dim, 1)),
        )

    def forward(self, states, actions):

        q = self.f(torch.cat([states, actions], dim=-1))
        return q
    

# class V_Net(nn.Module):
#     def __init__(self, state_dim, hidden_dim):
#         super(V_Net, self).__init__()
#         self.f = nn.Sequential(
#             layer_init(nn.Linear(state_dim, hidden_dim)),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(inplace=True),
#             layer_init(nn.Linear(hidden_dim, 1)),
#         )

#     def forward(self, states):

#         v = self.f(states)
#         return v
    
class SAC:
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 config,
                 device, 
                 offset: int):
        self.hidden_dim = config['hidden_dim']
        self.agent_number = config['agent_number']
        self.state_dim = state_dim
        self.actor = Policy(state_dim, self.hidden_dim, action_dim).to(device)
        self.critic_1 = QValueNet(state_dim, action_dim, self.hidden_dim, self.agent_number).to(device)
        self.critic_2 = QValueNet(state_dim, action_dim, self.hidden_dim, self.agent_number).to(device)
        # self.critic_v = V_Net(state_dim, self.hidden_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim,action_dim,self.hidden_dim, self.agent_number).to(device)
        self.target_critic_2 = QValueNet(state_dim,action_dim,self.hidden_dim, self.agent_number).to(device)
        # self.target_critic_v = V_Net(state_dim,self.hidden_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_lr = config['actor_learning_rate']
        self.critic_lr = config['critic_learning_rate']
        self.alpha_learning_rate = config['alpha_learning_rate']
        self.gamma = config['gamma']
        self.device = device
        self.offset = offset
        self.epochs = config['epochs']
        self.tau = config['tau']
        self.kld_weight = config['kld_weight']
        self.algorithm = config['algorithm']
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(),
                                                lr=self.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=self.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=self.critic_lr)
        # self.critic_v_optimizer = torch.optim.Adam(self.critic_v.parameters(),
        #                                            lr=self.critic_lr)

        self.log_alpha = torch.tensor(np.log(0.005), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=self.alpha_learning_rate)
        self.target_entropy = -action_dim
        
    def sample(self, transition, start_index):

        s = []
        a = []
        r = []
        s_next = []
        done = []

        with torch.no_grad():
            for row in range(start_index, start_index+4):
                s += transition[row]['states']
                a += transition[row]['actions']
                r += transition[row]['rewards']
                s_next += transition[row]['next_states']
                done += transition[row]['dones']

        return s, a, r, s_next, done
    

    def choose_action(self, state, scale):
        with torch.no_grad():
            state = torch.flatten(state,start_dim=0).unsqueeze(0)
            _,_,action, _ = self.actor(state, scale)
        return action
    
    def get_target(self, rewards, next_states, dones, scale):
        with torch.no_grad():
            _,_,actions, log_prob = self.actor(next_states, scale)
            entropy = -log_prob
            q1_value = self.target_critic_1(next_states, actions)
            q2_value = self.target_critic_2(next_states, actions)
            next_value = torch.min(q1_value,
                                  q2_value) + self.log_alpha.exp() * entropy

            td_target = rewards + self.gamma * next_value * (1 - dones)

        return td_target
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)
    
    def update(self, transition, scale, agent_index):
        
        for epoch in tqdm(range(self.epochs)):
            
            start_index = 0
            
            for i in range(8):

                states,  actions, rewards, next_states, dones= self.sample(transition, start_index)
                states = torch.stack(states, dim=0).flatten(start_dim=1)
                next_states = torch.stack(next_states, dim=0).flatten(start_dim=1)
                rewards = torch.stack(rewards, dim=0).view(-1,1)
                dones = torch.stack(dones, dim=0).view(-1,1)
                actions = torch.stack(actions, dim=0).flatten(start_dim=1)

                rewards = (rewards - rewards.mean())/(rewards.std() + 1e-8)

                td_target = self.get_target(rewards, next_states, dones, scale)
                critic_1_loss = torch.mean(
                    F.mse_loss(self.critic_1(states, actions), td_target.detach()))
                critic_2_loss = torch.mean(
                    F.mse_loss(self.critic_2(states, actions), td_target.detach()))

                _,_,new_actions, log_prob = self.actor(states, scale)
                
                entropy = -log_prob
                q1_value = self.critic_1(states, new_actions)
                q2_value = self.critic_2(states, new_actions)
                self.critic_1_optimizer.zero_grad()
                critic_1_loss.backward()
                self.critic_1_optimizer.step()
                self.critic_2_optimizer.zero_grad()
                critic_2_loss.backward()
                self.critic_2_optimizer.step()
                # vf_target = self.target_critic_v(next_states)
                # v_pred = self.critic_v(states)
                # v_target = torch.min(q1_value, q2_value) - self.log_alpha * log_prob
                # v_loss = F.mse_loss(v_pred, v_target.detach())

                # advantage = torch.min(q1_value, q2_value) - v_pred.detach()
                actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                            torch.min(q1_value, q2_value))
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # self.soft_update(self.critic_v, self.target_critic_v)
                
                # self.critic_v_optimizer.zero_grad()
                # v_loss.backward()
                # self.critic_v_optimizer.step()
                alpha_loss = torch.mean(
                (entropy - self.target_entropy).detach() * self.log_alpha.exp())
                self.log_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()
                
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic_1.parameters()) + list(self.critic_2.parameters()), 0.5)

                start_index += 4

                self.soft_update(self.critic_1, self.target_critic_1)
                self.soft_update(self.critic_2, self.target_critic_2)

        
