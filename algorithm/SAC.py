import torch
import math, random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Laplace
import numpy as np
from tqdm import tqdm
from utils.normalizing_flow import MADE, BatchNormFlow, Reverse, FlowSequential


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

        noise = Normal(scale, 0.1)
        mean = self.f(state) + noise.sample()
        mean = torch.cumsum(mean.reshape(-1,6,5,2), dim=-2)
        mean = mean[:, -1, :, :]
        mean = mean.flatten(start_dim = 1)

        b = self.f(state)
        b = torch.cumsum(F.elu_(b.reshape(-1,6,5,2),alpha = 1.0) + 1.0, dim=-2) + 0.1
        b = b[:, -1, :, :]
        b = b.flatten(start_dim = 1)

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
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QValueNet, self).__init__()
        self.f = nn.Sequential(
            layer_init(nn.Linear(state_dim+(action_dim//6), hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(hidden_dim, 1)),
        )

        self.linear = nn.Linear(3*4*12, 4*12)

    def forward(self, states, actions, agent_number):
        if agent_number > 1:
        
            states = self.linear(states.transpose(0,1))
            states = states.transpose(0,1)

        v = self.f(torch.cat([states, actions], dim=-1))
        return v
    
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
        self.critic_1 = QValueNet(state_dim, action_dim, self.hidden_dim).to(device)
        self.critic_2 = QValueNet(state_dim, action_dim, self.hidden_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim,action_dim,self.hidden_dim).to(device)
        self.target_critic_2 = QValueNet(state_dim,action_dim,self.hidden_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_lr = config['actor_learning_rate']
        self.critic_lr = config['critic_learning_rate']
        self.density_lr = config['density_learning_rate']
        self.alpha_learning_rate = config['alpha_learning_rate']
        self.gamma = config['gamma']
        self.device = device
        self.offset = offset
        self.epochs = config['epochs']
        self.tau = config['tau']
        self.beta_coef = config['beta_coef']
        self.kld_weight = config['kld_weight']
        self.algorithm = config['algorithm']
        self.kl_coef = config['kl_coef']
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(),
                                                lr=self.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=self.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=self.critic_lr)

        self.log_alpha = torch.tensor(np.log(0.005), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=self.alpha_learning_rate)
        self.target_entropy = -action_dim

        self._init_density_model(state_dim, self.hidden_dim)
        
    def _init_density_model(self, state_dim, hidden_dim):
        # Creat density model
        modules = []
        for i in range(2):
            modules += [
                MADE(num_inputs=state_dim,
                     num_hidden=hidden_dim,
                     num_cond_inputs=None,
                     ),
                BatchNormFlow(state_dim, ),
                Reverse(state_dim, )
            ]
        model = FlowSequential(*modules)

        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)
        model.to(self.device)
        self.density_model = model
        self.density_optimizer = torch.optim.Adam(self.density_model.parameters(), lr=self.density_lr)
        
    def sample(self, transition, buffer_batchsize, start_index):

        s = []
        a = []
        r = []
        s_next = []
        done = []
        ground_b = []

        with torch.no_grad():
            for row in range(start_index, start_index+buffer_batchsize//8):
                s += transition[row]['states']
                a += transition[row]['actions']
                r += transition[row]['rewards']
                s_next += transition[row]['next_states']
                done += transition[row]['dones']
                ground_b += transition[row]['ground_truth_b']

        return s, a, r, s_next, done, ground_b
    

    def choose_action(self, state, scale):
        with torch.no_grad():
            state = torch.flatten(state,start_dim=0).unsqueeze(0)
            _,_,action, _ = self.actor(state, scale)
        return action
    
    def get_target(self, rewards, next_states, dones, scale):
        with torch.no_grad():
            _,_,actions, log_prob = self.actor(next_states, scale)
            entropy = -log_prob.unsqueeze(-1)
            next_Q_input = torch.cat([next_states, actions], dim=-1)
            q1_value = self.target_critic_1(next_Q_input)
            q2_value = self.target_critic_2(next_Q_input)
            next_value = torch.min(q1_value,
                                  q2_value) + self.log_alpha.exp() * entropy
            td_target = rewards + self.gamma * next_value * (1 - dones)
            _, log_prob_game = self.density_model.log_probs(inputs=next_states,
                                                                        cond_inputs=None)
            log_prob_game = F.sigmoid((log_prob_game - log_prob_game.mean()) / log_prob_game.std())
            beta_t = self.beta_coef / (log_prob_game)
            
            td_target = td_target + beta_t

        return td_target
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)
    
    def update(self, transition, buffer_batchsize, scale):

        for _ in range(50):
                        
                for row in range(buffer_batchsize):
                
                    nominal_data_batch = transition[row]['states']
                    nominal_data_batch = torch.stack(nominal_data_batch, dim=0).flatten(start_dim=1)

                    m_loss, log_prob = self.density_model.log_probs(inputs=nominal_data_batch,
                                                                    cond_inputs=None)
                    self.density_optimizer.zero_grad()
                    density_loss = -m_loss.mean()
                    density_loss.backward()
                    self.density_optimizer.step()
        
        for epoch in tqdm(range(self.epochs)):

            start_index = 0

            for _ in range(8):

                states,  actions, rewards, next_states, dones, ground_b= self.sample(transition, buffer_batchsize, start_index)
                states = torch.stack(states, dim=0).flatten(start_dim=1)
                next_states = torch.stack(next_states, dim=0).flatten(start_dim=1)
                rewards = torch.stack(rewards, dim=0).view(-1,1)
                dones = torch.stack(dones, dim=0).view(-1,1)
                ground_b = torch.stack(ground_b, dim=0).flatten(start_dim=1)
                actions = torch.stack(actions, dim=0).flatten(start_dim=1)
                rewards = rewards / (rewards.std() + 1e-5)

                Q_input = torch.cat([states, actions], dim=-1)
                td_target = self.get_target(rewards, next_states, dones, scale)
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

                mean, b, new_actions, log_prob = self.actor(states, scale)
                entropy = -log_prob
                new_Q_input = torch.cat([states, new_actions], dim=-1)
                q1_value = self.critic_1(new_Q_input)
                q2_value = self.critic_2(new_Q_input)
                actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                        torch.min(q1_value, q2_value))
                KL = (ground_b*torch.exp((-torch.abs(mean - actions)/ground_b))+torch.abs(mean - actions))/b + torch.log(b/ground_b) - 1
                actor_loss = actor_loss - torch.mean(self.kl_coef*KL)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                alpha_loss = torch.mean(
                (entropy - self.target_entropy).detach() * self.log_alpha.exp())
                self.log_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic_1.parameters()) + list(self.critic_2.parameters()), 0.5)

                self.soft_update(self.critic_1, self.target_critic_1)
                self.soft_update(self.critic_2, self.target_critic_2)

                start_index += buffer_batchsize//8

        
