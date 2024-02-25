from algorithm.ppo import PPO
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Laplace
import numpy as np
from tqdm import tqdm

class MAPPO(PPO):
    
    def sample(self, transition, start_index, agent_index):

        s = []
        o = []
        a = []
        r = []
        s_next = []
        done = []

        with torch.no_grad():

                for row in range(start_index, start_index+4):
                    for i in range(self.agent_number):
                        s+=transition[row]['states'][i]
                        
                        s_next+=transition[row]['next_states'][i]
                    a+=transition[row]['actions'][agent_index]
                    r+=transition[row]['rewards'][agent_index]
                    o += transition[row]['states'][agent_index]
              
                    done+=transition[row]['dones']
                return s, o, a, r, s_next, done

    def update(self, transition, buffer_batchsize, scale, agent_index):
        
        for epoch in tqdm(range(self.epochs)):
            
            start_index = 0
            
            for _ in range(8):

                states, observations, actions, rewards, next_states,dones= self.sample(transition, start_index, agent_index)

                dones = torch.stack(dones, dim=0).view(-1,1)
                states = torch.stack(states, dim=0).reshape(-1,self.agent_number*self.state_dim).type(torch.FloatTensor).to(self.device)
                next_states = torch.stack(next_states, dim=0).reshape(-1,self.agent_number*self.state_dim).type(torch.FloatTensor).to(self.device)
                observations = torch.stack(observations, dim=0).type(torch.FloatTensor).to(self.device)

                rewards = torch.stack(rewards, dim=0).view(-1,1).type(torch.FloatTensor).to(self.device)

                actions = torch.stack(actions, dim=0).flatten(start_dim=1).reshape(-1, 2*5).type(torch.FloatTensor).to(self.device)

                rewards = (rewards - rewards.mean())/(rewards.std() + 1e-8)

                # td_error
                with torch.no_grad():
                    next_state_value = self.value(next_states)
                    td_target = rewards + self.gamma * next_state_value * (1-dones)

                    td_value = self.value(states)
                    td_delta = td_target - td_value
                
                    # calculate GAE
                    advantage = 0
                    advantage_list = []
                    td_delta = td_delta.cpu().detach().numpy()
                    for delta in td_delta[::-1]:
                        advantage = self.gamma * self.lamda * advantage + delta
                        advantage_list.append(advantage)
                    advantage_list.reverse()
                    advantage = torch.tensor(np.array(advantage_list), dtype=torch.float).to(self.device).reshape(-1,1)
                    #advantage_normalization
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
                    # get ratio
                    mean, b = self.pi(observations, scale)
                    old_policy = Laplace(mean, b)
                    old_log_probs = old_policy.log_prob(actions)

                mean, b = self.pi(observations, scale)
                new_policy = Laplace(mean, b)
                log_probs = new_policy.log_prob(actions)
                ratio = torch.exp(log_probs - old_log_probs)
        
                # clipping
                ratio = ratio.flatten(start_dim=1)
                surr1 = ratio * advantage
                surr2 = advantage * torch.clamp(ratio, 1-self.eps, 1+self.eps)

                value = self.value(states)
                
                # pi and value loss
                pi_loss = torch.mean(-torch.min(surr1, surr2))
                value_loss = torch.mean(F.mse_loss(value, td_target.detach()))

                total_loss = (pi_loss + 2*value_loss - new_policy.entropy() * self.entropy_coef)
                # total_loss = pi_loss + value_loss

                self.optimizer.zero_grad()
                total_loss.mean().backward()
                nn.utils.clip_grad_norm_(self.pi.parameters(), 10)
                self.optimizer.step()

                if epoch%2 == 0:
                    self.old_pi.load_state_dict(self.pi.state_dict())
                # self.old_value.load_state_dict(self.value.state_dict())
                start_index += 4
        
