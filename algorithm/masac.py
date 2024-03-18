import torch
import math, random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace
from torch.distributions.kl import kl_divergence
import numpy as np
from tqdm import tqdm
from algorithm.SAC import SAC
from utils.normalizing_flow import MADE, BatchNormFlow, Reverse, FlowSequential

class MASAC(SAC):
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 config,
                 device, 
                 offset: int):
        super(MASAC, self).__init__(state_dim, action_dim, config, device, offset)
        if self.algorithm == 'CCE-MASAC':
            self.density_lr = config['density_learning_rate']
            self.beta_coef = config['beta_coef']
            self.eta_coef = config['eta_coef']
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
    
    def sample(self, transition, start_index, agent_index):

        s = []
        o = []
        a = []
        r = []
        s_next = []
        o_next = []
        done = []
        ground_b = []

        with torch.no_grad():
                # for _ in range(sample_batchsize):
                for row in range(start_index, start_index+self.mini_batch):
                    for i in range(self.agent_number):
                        s+=transition[row]['states'][i]
                        
                        s_next+=transition[row]['next_states'][i]
                    a+=transition[row]['actions'][agent_index]
                    r+=transition[row]['rewards'][agent_index]
                    o += transition[row]['states'][agent_index]
                    o_next += transition[row]['next_states'][agent_index]
                    ground_b += transition[row]['ground_b'][agent_index]
                    done+=transition[row]['dones']
                return s, o, a, r, o_next, s_next, done, ground_b
    
    def get_target(self, rewards, next_states, next_observations, dones, scale, extra_signal):
        with torch.no_grad():
            _,_,actions, log_prob = self.actor(next_observations, scale)
            entropy = -log_prob
            q1_value = self.target_critic_1(next_states, actions)
            q2_value = self.target_critic_2(next_states, actions)
            next_value = torch.min(q1_value,
                                  q2_value) + self.log_alpha.exp() * entropy

            
            if self.algorithm == 'CCE-MASAC':
                td_target = rewards + extra_signal.mean(dim=1, keepdim=True) + self.gamma * next_value * (1 - dones)
                _, log_prob_game = self.density_model.log_probs(inputs=next_observations,
                                                                            cond_inputs=None)
                log_prob_game = F.sigmoid(((log_prob_game - log_prob_game.mean()) / log_prob_game.std()).exp())
                beta_t = self.beta_coef / log_prob_game
                
                td_target = td_target + beta_t
            else:
                td_target = rewards + self.gamma * next_value * (1 - dones)

        return td_target
    
    def _init_density_model(self, state_dim, hidden_dim):
        # Creat density model
        modules = []
        for i in range(3):
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
    
    def update(self, transition, buffer_batchsize, scale, agent_index):
            
        
        if self.algorithm == 'CCE-MASAC':
            for _ in range(self.epochs):
                            
                    for row in range(buffer_batchsize):
                    
                        nominal_data_batch = transition[row]['states'][agent_index]
                        nominal_data_batch = torch.stack(nominal_data_batch, dim=0).flatten(start_dim=1)

                        m_loss, log_prob = self.density_model.log_probs(inputs=nominal_data_batch,
                                                                        cond_inputs=None)
                        self.density_optimizer.zero_grad()
                        density_loss = -m_loss.mean()
                        density_loss.backward()
                        self.density_optimizer.step()

        extra_signal = torch.zeros(self.mini_batch*len(transition[0]['states'][0]), 1).to(self.device)
        temp_pi_old_list = []
        
        for epoch in tqdm(range(self.epochs)):
            
            start_index = 0
            pi_old_list = []
            
            for i in range(int(buffer_batchsize/self.mini_batch)):
                states, observations, actions, rewards, next_observations, next_states, dones, ground_b = self.sample(transition, start_index, agent_index)

                dones = torch.stack(dones, dim=0).view(-1,1)
                states = torch.stack(states, dim=0).reshape(-1,self.agent_number*self.state_dim).type(torch.FloatTensor).to(self.device)
                next_states = torch.stack(next_states, dim=0).reshape(-1,self.agent_number*self.state_dim).type(torch.FloatTensor).to(self.device)
                observations = torch.stack(observations, dim=0).type(torch.FloatTensor).to(self.device)
                next_observations = torch.stack(next_observations, dim=0).type(torch.FloatTensor).to(self.device)

                rewards = torch.stack(rewards, dim=0).view(-1,1).type(torch.FloatTensor).to(self.device)

                actions = torch.stack(actions, dim=0).flatten(start_dim=1).reshape(-1, 2*5).type(torch.FloatTensor).to(self.device)
                ground_b = torch.stack(ground_b, dim=0).flatten(start_dim=1).reshape(-1, 2*5).type(torch.FloatTensor).to(self.device)

                rewards = (rewards - rewards.mean())/(rewards.std() + 1e-8)

                td_target = self.get_target(rewards, next_states, next_observations, dones, scale, extra_signal)
                critic_1_loss = torch.mean(
                    F.mse_loss(self.critic_1(states, actions), td_target.detach()))
                critic_2_loss = torch.mean(
                    F.mse_loss(self.critic_2(states, actions), td_target.detach()))

                mean,b,new_actions, log_prob_pi = self.actor(observations, scale)
                if self.algorithm == 'CCE-MASAC':
                    pi_old_list.append(Laplace(mean, b))
                
                entropy = -log_prob_pi
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
                if epoch == 0 or self.algorithm == 'MASAC':
                    actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                            torch.min(q1_value, q2_value))
                elif epoch > 0 and self.algorithm == 'CCE-MASAC':
                    dist1 = Laplace(actions, ground_b)
                    extra_signal1 = dist1.log_prob(dist1.sample())
                    dist2 = temp_pi_old_list[i]
                    extra_signal2 = dist2.log_prob(dist2.sample())
                    extra_signal = self.eta_coef * (extra_signal1 + extra_signal2)
                    # actor_loss = torch.mean(-self.log_alpha.exp() * entropy-
                    #                         torch.min(q1_value, q2_value) - extra_signal)
                    actor_loss = torch.mean(-self.log_alpha.exp() * (- torch.mean(kl_divergence(Laplace(mean,b), Laplace(actions, ground_b)))) -
                                            torch.min(q1_value, q2_value) + self.eta_coef * torch.mean(kl_divergence(Laplace(mean,b), temp_pi_old_list[i]))) 
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
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

                start_index += self.mini_batch

                self.soft_update(self.critic_1, self.target_critic_1)
                self.soft_update(self.critic_2, self.target_critic_2)
            
                # if epoch == self.epochs - 1:
                #     value = torch.min(q1_value,
                #                   q2_value) + self.log_alpha.exp() * entropy
                #     v_list = np.append(v_list,  float(torch.mean(value.squeeze(-1))))
        
            temp_pi_old_list = pi_old_list
        # return float(v_list.mean())



        
