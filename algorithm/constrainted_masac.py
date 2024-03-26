import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace
from torch.distributions.kl import kl_divergence
import numpy as np
from tqdm import tqdm
from algorithm.masac import MASAC
from utils.dual_variable import DualVariable
from utils.normalizing_flow import MADE, BatchNormFlow, Reverse, FlowSequential

class Constrained_MASAC(MASAC):
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 config,
                 device, 
                 offset: int):
        super(Constrained_MASAC, self).__init__(state_dim, action_dim, config, device, offset)
        self.budget: float = 0.0
        self.penalty_learning_rate: float = 0.01,
        self.penalty_min_value: Optional[float] = None,
        self.penalty_initial_value: float = 1,
        self.dual = DualVariable(self.budget, self.penalty_learning_rate, self.penalty_initial_value, self.penalty_min_value)
    
    def sample(self, transition, start_index, agent_index):

        s = []
        o = []
        a = []
        r = []
        c = []
        s_next = []
        o_next = []
        done = []

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
                    done+=transition[row]['dones']
                return s, o, a, r, c, o_next, s_next, done
    
    def get_target(self, rewards, costs, next_states, next_observations, dones, noise):
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

            td_target = rewards + self.gamma * next_value * (1 - dones)
            td_target_cost = costs + self.gamma * next_value_cost * (1 - dones)

        return td_target, td_target_cost

    def update(self, transition, buffer_batchsize, noise, agent_index):
            
        
        for epoch in tqdm(range(self.epochs)):
            
            start_index = 0
            
            for i in range(int(buffer_batchsize/self.mini_batch)):
                states, observations, actions, rewards, costs, next_observations, next_states, dones = self.sample(transition, start_index, agent_index)

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

                next_q_values_target, next_q_values_target_cost = self.get_target(rewards, costs, next_states, next_observations, dones, noise)

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
                
                entropy = -log_prob_pi
                q1_value = self.critic_1(states, new_actions)
                q2_value = self.critic_2(states, new_actions)
                q1_value_cost = self.critic_1_cost(states, new_actions)
                q2_value_cost = self.critic_2_cost(states, new_actions)

                actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                            torch.min(q1_value, q2_value))

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