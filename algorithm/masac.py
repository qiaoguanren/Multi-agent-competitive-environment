import torch
import math, random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from tqdm import tqdm
from algorithm.SAC import SAC
from utils.normalizing_flow import MADE, BatchNormFlow, Reverse, FlowSequential

class MASAC(SAC):
    
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
                for row in range(start_index, start_index+4):
                    for i in range(self.agent_number):
                        s+=transition[row]['states'][i]
                        
                        s_next+=transition[row]['next_states'][i]
                    a+=transition[row]['actions'][agent_index]
                    r+=transition[row]['rewards'][agent_index]
                    o+=transition[row]['states'][agent_index]
                    o_next+=transition[row]['next_states'][agent_index]
              
                    done+=transition[row]['dones']
                    ground_b+=transition[row]['ground_b'][agent_index]
                return s, o, a, r, s_next,o_next, done, ground_b
    
    def get_target(self, rewards, next_states, next_conservations, dones, scale):
        with torch.no_grad():
            _,_,actions, log_prob = self.actor(next_conservations, scale)
            entropy = -log_prob
            q1_value = self.target_critic_1(next_states, actions, self.agent_number)
            q2_value = self.target_critic_2(next_states, actions, self.agent_number)
            next_value = torch.min(q1_value,
                                  q2_value) + self.log_alpha.exp() * entropy

            td_target = rewards + self.gamma * next_value * (1 - dones)

        return td_target
    
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
    
    def update(self, transition, buffer_batchsize, scale, agent_index):

        # for _ in range(50):
                        
        #         for row in range(buffer_batchsize):
                
        #             nominal_data_batch = transition[row]['states'][agent_index]
        #             nominal_data_batch = torch.stack(nominal_data_batch, dim=0).flatten(start_dim=1)

        #             m_loss, log_prob = self.density_model.log_probs(inputs=nominal_data_batch,
        #                                                             cond_inputs=None)
        #             self.density_optimizer.zero_grad()
        #             density_loss = -m_loss.mean()
        #             density_loss.backward()
        #             self.density_optimizer.step()

        v_list = np.array([])
        
        for epoch in tqdm(range(self.epochs)):
            
            start_index = 0
            
            for _ in range(8):

                states, observations, actions, rewards, next_states,next_observations, dones, ground_b= self.sample(transition, start_index, agent_index)
                    
                observations = torch.stack(observations, dim=0).flatten(start_dim=1).type(torch.FloatTensor).to(self.device)
                next_observations = torch.stack(next_observations, dim=0).flatten(start_dim=1).type(torch.FloatTensor).to(self.device)

                dones = torch.stack(dones, dim=0).view(-1,1)
                states = torch.stack(states, dim=0).flatten(start_dim=1).type(torch.FloatTensor).to(self.device)
                next_states = torch.stack(next_states, dim=0).flatten(start_dim=1).type(torch.FloatTensor).to(self.device)
                ground_b = torch.stack(ground_b, dim=0).flatten(start_dim=1).type(torch.FloatTensor).to(self.device)
                rewards = torch.stack(rewards, dim=0).view(-1,1).type(torch.FloatTensor).to(self.device)

                actions = torch.stack(actions, dim=0).flatten(start_dim=1).type(torch.FloatTensor).to(self.device)

                rewards = (rewards - rewards.mean())/(rewards.std() + 1e-8)

                td_target = self.get_target(rewards, next_states, next_observations, dones, scale)
                critic_1_loss = torch.mean(
                    F.mse_loss(self.critic_1(states, actions, self.agent_number), td_target.detach()))
                critic_2_loss = torch.mean(
                    F.mse_loss(self.critic_2(states, actions, self.agent_number), td_target.detach()))
                self.critic_1_optimizer.zero_grad()
                critic_1_loss.backward()
                self.critic_1_optimizer.step()
                self.critic_2_optimizer.zero_grad()
                critic_2_loss.backward()
                self.critic_2_optimizer.step()

                _,_,new_actions, log_prob = self.actor(observations, scale)
                entropy = -log_prob
                q1_value = self.critic_1(states, new_actions, self.agent_number)
                q2_value = self.critic_2(states, new_actions, self.agent_number)
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
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic_1.parameters()) + list(self.critic_2.parameters()), 0.5)


                self.soft_update(self.critic_1, self.target_critic_1)
                self.soft_update(self.critic_2, self.target_critic_2)

                start_index += 4
            
                if epoch == self.epochs - 1:
                    v_list = np.append(v_list,  float(torch.mean(q1_value.squeeze(-1))))
        
        return float(v_list.mean())



        
