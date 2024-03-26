import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace
from torch.distributions.kl import kl_divergence
import numpy as np
from tqdm import tqdm
from algorithm.SAC import SAC

class MASAC(SAC):
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 config,
                 device, 
                 offset: int):
        super(MASAC, self).__init__(state_dim, action_dim, config, device, offset)

    def choose_action(self, state, noise):
        with torch.no_grad():
            state = torch.flatten(state,start_dim=0).unsqueeze(0)
            mean, scale,action, log_prob = self.actor(state, noise)
        return mean, scale, action, log_prob
    
    def sample(self, transition, start_index, agent_index):

        s = []
        o = []
        a = []
        r = []
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
                    o += transition[row]['states'][agent_index]
                    o_next += transition[row]['next_states'][agent_index]
                    done+=transition[row]['dones']
                return s, o, a, r, o_next, s_next, done
    
    def get_target(self, rewards, next_states, next_observations, dones, noise):
        with torch.no_grad():
            _,_,actions, log_prob = self.actor(next_observations, noise)
            entropy = -log_prob
            q1_value = self.target_critic_1(next_states, actions)
            q2_value = self.target_critic_2(next_states, actions)

            next_value = torch.min(q1_value,
                                  q2_value) + self.log_alpha.exp() * entropy

            td_target = rewards + self.gamma * next_value * (1 - dones)

        return td_target
    
    def update(self, transition, buffer_batchsize, noise, agent_index):
        
        for epoch in tqdm(range(self.epochs)):
            
            start_index = 0
            
            for i in range(int(buffer_batchsize/self.mini_batch)):
                states, observations, actions, rewards, next_observations, next_states, dones = self.sample(transition, start_index, agent_index)

                dones = torch.stack(dones, dim=0).view(-1,1)
                states = torch.stack(states, dim=0).reshape(-1,self.agent_number*self.state_dim).type(torch.FloatTensor).to(self.device)
                next_states = torch.stack(next_states, dim=0).reshape(-1,self.agent_number*self.state_dim).type(torch.FloatTensor).to(self.device)
                observations = torch.stack(observations, dim=0).type(torch.FloatTensor).flatten(start_dim=1).to(self.device)
                next_observations = torch.stack(next_observations, dim=0).type(torch.FloatTensor).flatten(start_dim=1).to(self.device)

                rewards = torch.stack(rewards, dim=0).view(-1,1).type(torch.FloatTensor).to(self.device)

                actions = torch.stack(actions, dim=0).flatten(start_dim=1).reshape(-1, 3*5).type(torch.FloatTensor).to(self.device)

                rewards = (rewards - rewards.mean())/(rewards.std() + 1e-8)

                next_q_values_target = self.get_target(rewards, next_states, next_observations, dones, noise)

                critic_1_loss = torch.mean(
                    F.mse_loss(self.critic_1(states, actions), next_q_values_target.detach()))
                critic_2_loss = torch.mean(
                    F.mse_loss(self.critic_2(states, actions), next_q_values_target.detach()))
                
                self.critic_1_optimizer.zero_grad()
                critic_1_loss.backward()
                self.critic_1_optimizer.step()
                self.critic_2_optimizer.zero_grad()
                critic_2_loss.backward()
                self.critic_2_optimizer.step()

                # torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=10.0, norm_type=2)
                # torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=10.0, norm_type=2)
                # torch.nn.utils.clip_grad_norm_(self.critic_1_cost.parameters(), max_norm=10.0, norm_type=2)
                # torch.nn.utils.clip_grad_norm_(self.critic_2_cost.parameters(), max_norm=10.0, norm_type=2)

                _, _, new_actions, log_prob_pi = self.actor(observations, noise)
                
                entropy = -log_prob_pi
                q1_value = self.critic_1(states, new_actions)
                q2_value = self.critic_2(states, new_actions)

                actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                            torch.min(q1_value, q2_value))

                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
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
    



        
