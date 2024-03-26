import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace
from torch.distributions.kl import kl_divergence
import numpy as np
from tqdm import tqdm
from algorithm.constrainted_masac import Constrained_MASAC
from utils.dual_variable import DualVariable
from utils.normalizing_flow import MADE, BatchNormFlow, Reverse, FlowSequential

class Constrainted_CCE_MASAC(Constrained_MASAC):
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 config,
                 device, 
                 offset: int):
        super(Constrainted_CCE_MASAC, self).__init__(state_dim, action_dim, config, device, offset)

        self.density_lr = config['density_learning_rate']
        self.beta_coef = config['beta_coef']
        self.eta_coef1 = config['eta_coef1']
        self.eta_coef2 = config['eta_coef2']
        self._init_density_model(state_dim, self.hidden_dim)
        
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

    
    def sample(self, transition, start_index, agent_index):

        s = []
        o = []
        a = []
        r = []
        c = []
        s_next = []
        o_next = []
        done = []
        magnet = 0

        with torch.no_grad():
                # for _ in range(sample_batchsize):
                for row in range(start_index, start_index+self.mini_batch):
                    for i in range(self.agent_number):
                        s+=transition[row]['states'][i]
                        
                        s_next+=transition[row]['next_states'][i]
                    a+=transition[row]['actions'][agent_index]
                    r+=transition[row]['rewards'][agent_index]
                    c+=transition[row]['costs'][agent_index]
                    o += transition[row]['states'][agent_index]
                    o_next += transition[row]['next_states'][agent_index]
                    magnet += sum(transition[row]['magnet'][agent_index])/12
                    done+=transition[row]['dones']
                return s, o, a, r, c, o_next, s_next, done, magnet/self.mini_batch
    
    def get_target(self, rewards, costs, next_states, next_observations, dones, noise, signal):
        with torch.no_grad():
            _,_,actions, log_prob = self.actor(next_observations, noise)
            entropy = -log_prob
            q1_value = self.target_critic_1(next_states, actions)
            q2_value = self.target_critic_2(next_states, actions)
            q1_value_cost = self.target_critic_1_cost(next_states, actions)
            q2_value_cost = self.target_critic_2_cost(next_states, actions)

            next_value = torch.min(q1_value,
                                  q2_value) + self.log_alpha.exp() * entropy
            next_value_cost = torch.min(q1_value_cost,
                                  q2_value_cost) + self.log_alpha.exp() * entropy


            td_target = rewards + signal.mean(dim=1, keepdim=True) + self.gamma * next_value * (1 - dones)
            _, log_prob_game = self.density_model.log_probs(inputs=next_observations,
                                                                        cond_inputs=None)
            log_prob_game = F.sigmoid(((log_prob_game - log_prob_game.mean()) / log_prob_game.std()).exp())
            beta_t = self.beta_coef / log_prob_game
            
            td_target = td_target + beta_t
            td_target_cost = costs + self.gamma * next_value_cost * (1 - dones)

        return td_target, td_target_cost
    
    def update(self, transition, buffer_batchsize, noise, agent_index):
            
        for _ in range(self.epochs):
                        
                for row in range(buffer_batchsize):
                
                    nominal_data_batch = transition[row]['states'][agent_index]
                    nominal_data_batch = torch.stack(nominal_data_batch, dim=0).flatten(start_dim=1)

                    m_loss, _ = self.density_model.log_probs(inputs=nominal_data_batch,
                                                                    cond_inputs=None)
                    self.density_optimizer.zero_grad()
                    density_loss = -m_loss.mean()
                    density_loss.backward()
                    self.density_optimizer.step()

        signal = torch.zeros(self.mini_batch*len(transition[0]['states'][0]), 1).to(self.device)
        temp_pi_old_list = []
        
        for epoch in tqdm(range(self.epochs)):
            
            start_index = 0
            pi_old_list = []
            
            for i in range(int(buffer_batchsize/self.mini_batch)):
                states, observations, actions, rewards, costs, next_observations, next_states, dones, magnet = self.sample(transition, start_index, agent_index)

                dones = torch.stack(dones, dim=0).view(-1,1)
                states = torch.stack(states, dim=0).reshape(-1,self.agent_number*self.state_dim).type(torch.FloatTensor).to(self.device)
                next_states = torch.stack(next_states, dim=0).reshape(-1,self.agent_number*self.state_dim).type(torch.FloatTensor).to(self.device)
                observations = torch.stack(observations, dim=0).type(torch.FloatTensor).flatten(start_dim=1).to(self.device)
                next_observations = torch.stack(next_observations, dim=0).type(torch.FloatTensor).flatten(start_dim=1).to(self.device)

                rewards = torch.stack(rewards, dim=0).view(-1,1).type(torch.FloatTensor).to(self.device)
                costs = torch.stack(costs, dim=0).view(-1,1).type(torch.FloatTensor).to(self.device)

                actions = torch.stack(actions, dim=0).flatten(start_dim=1).reshape(-1, 3*5).type(torch.FloatTensor).to(self.device)

                rewards = (rewards - rewards.mean())/(rewards.std() + 1e-8)
                costs = (costs - costs.mean())/(costs.std() + 1e-8)

                next_q_values_target, next_q_values_target_cost = self.get_target(rewards, costs, next_states, next_observations, dones, noise, signal)

                critic_1_loss = torch.mean(
                    F.mse_loss(self.critic_1(states, actions), next_q_values_target.detach()))
                critic_2_loss = torch.mean(
                    F.mse_loss(self.critic_2(states, actions), next_q_values_target.detach()))
                critic_1_loss_cost = torch.mean(
                    F.mse_loss(self.critic_1_cost(states, actions), next_q_values_target_cost.detach()))
                critic_2_loss_cost = torch.mean(
                    F.mse_loss(self.critic_2_cost(states, actions), next_q_values_target_cost.detach()))
                
                self.critic_1_optimizer.zero_grad()
                critic_1_loss.backward()
                self.critic_1_optimizer.step()
                self.critic_2_optimizer.zero_grad()
                critic_2_loss.backward()
                self.critic_2_optimizer.step()
                self.critic_1_cost_optimizer.zero_grad()
                critic_1_loss_cost.backward()
                self.critic_1_cost_optimizer.step()
                self.critic_2_cost_optimizer.zero_grad()
                critic_2_loss_cost.backward()
                self.critic_2_cost_optimizer.step()

                # torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=10.0, norm_type=2)
                # torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=10.0, norm_type=2)
                # torch.nn.utils.clip_grad_norm_(self.critic_1_cost.parameters(), max_norm=10.0, norm_type=2)
                # torch.nn.utils.clip_grad_norm_(self.critic_2_cost.parameters(), max_norm=10.0, norm_type=2)

                _, _, new_actions, log_prob_pi = self.actor(observations, noise)
                pi_old_list.append(log_prob_pi)
                
                entropy = -log_prob_pi
                q1_value = self.critic_1(states, new_actions)
                q2_value = self.critic_2(states, new_actions)
                q1_value_cost = self.critic_1_cost(states, new_actions)
                q2_value_cost = self.critic_2_cost(states, new_actions)

                if epoch == 0:
                    actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                            torch.min(q1_value, q2_value))
                else:
                    signal1 = magnet
                    signal2 = temp_pi_old_list[i]
                    signal = self.eta_coef1 * signal1 + self.eta_coef2 * signal2
                    actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                            torch.min(q1_value, q2_value) - signal)

                current_penalty = self.dual.nu().item()
                penalty = current_penalty * torch.min(q1_value_cost, q2_value_cost)
                actor_lagrangian_loss = (actor_loss + penalty).sum(dim=1).mean()
                self.actor_optimizer.zero_grad()
                actor_lagrangian_loss.backward(retain_graph=True)
                self.actor_optimizer.step()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5, norm_type=2)
                
                alpha_loss = torch.mean(
                (entropy - self.target_entropy).detach() * self.log_alpha.exp())
                self.log_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()

                start_index += self.mini_batch

                self.soft_update(self.critic_1, self.target_critic_1)
                self.soft_update(self.critic_2, self.target_critic_2)
                self.soft_update(self.critic_1_cost, self.target_critic_1_cost)
                self.soft_update(self.critic_2_cost, self.target_critic_2_cost)
        
            temp_pi_old_list = pi_old_list