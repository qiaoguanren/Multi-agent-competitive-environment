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
        b = []

        if self.algorithm == 'IPPO':
            with torch.no_grad():
                for row in range(start_index, start_index+4):
                # for _ in range(offset):
                    r.append(transition[row]['rewards'][agent_index])
                    s_next.append(transition[row]['next_states'][agent_index])
                    a.append(transition[row]['actions'][agent_index])
                    o.append(transition[row]['states'][agent_index])
                    done.append(transition[row]['dones'])
                    b.append(transition[row]['ground_b'][agent_index])
            return o, a, r, s_next, done

        else:
            with torch.no_grad():
                # for _ in range(sample_batchsize):
                for row in range(start_index, start_index+4):
                    # row = random.randint(0,buffer_batchsize-1)
                    # col = random.randint(0,int(60/offset)-1)
                    # for i in range(self.agent_number):
                    #     s.append(transition[row]['states'][i][col])
                    #     r.append(transition[row]['rewards'][i][col])
                    #     s_next.append(transition[row]['next_states'][i][col])
                    # a.append(transition[row]['actions'][agent_index][col])
                    # o.append(transition[row]['states'][agent_index][col])
                    for i in range(self.agent_number):
                        s.append(transition[row]['states'][i])
                        r.append(transition[row]['rewards'][i])
                        s_next.append(transition[row]['next_states'][i])
                    a.append(transition[row]['actions'][agent_index])
                    o.append(transition[row]['states'][agent_index])
                    b.append(transition[row]['ground_b'][agent_index])
              
                    done.append(transition[row]['dones'])
                return s, o, a, r, s_next, done, b

    def update(self, transition, buffer_batchsize, scale, agent_index):

        if self.algorithm == 'CCE-MAPPO':
            for _ in range(50):
                        
                for row in range(buffer_batchsize):
                
                    nominal_data_batch = transition[row]['states'][agent_index]
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

                if self.algorithm == 'IPPO':
                    states, actions, rewards, next_states, dones= self.sample(transition, start_index, agent_index)
                else:
                    states, observations, actions, rewards, next_states, dones, ground_b= self.sample(transition, start_index, agent_index)
                    ground_b = torch.stack(ground_b, dim=0).flatten(start_dim=1)
                    observations = torch.stack(observations, dim=0).flatten(start_dim=1)
                dones = torch.stack(dones, dim=0).view(-1,1)
                states = torch.stack(states, dim=0).flatten(start_dim=1)
                next_states = torch.stack(next_states, dim=0).flatten(start_dim=1)
                rewards = torch.stack(rewards, dim=0).view(-1,1)
                rewards = rewards / (rewards.std() + 1e-5)

                # td_error
                with torch.no_grad():
                    next_state_value = self.value(next_states)
                    next_state_value = (next_state_value - next_state_value.mean()) / (next_state_value.std() + 1e-5)
                    td_target = rewards + self.gamma * next_state_value * (1-dones)
                    
                    if self.algorithm == 'CCE-MAPPO':
                        _, log_prob_game = self.density_model.log_probs(inputs=next_states,
                                                                        cond_inputs=None)
                        log_prob_game = F.sigmoid((log_prob_game - log_prob_game.mean()) / log_prob_game.std())
                        beta_t = self.beta_coef / (log_prob_game)
                        
                        td_target = td_target + beta_t
                    td_value = self.value(states)
                    td_value = (td_value - td_value.mean()) / (td_value.std() + 1e-5)
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
                    if self.algorithm == 'IPPO':
                        mean, b = self.pi(states, scale)
                    else:
                        mean, b = self.pi(observations, scale)
                    old_policy = Laplace(mean, b)
                    old_log_probs = old_policy.log_prob(actions)

                if self.algorithm == 'IPPO':
                    mean, b = self.pi(states, scale)
                else:
                    mean, b = self.pi(observations, scale)
                new_policy = Laplace(mean, b)
                log_probs = new_policy.log_prob(actions)
                ratio = torch.exp(log_probs - old_log_probs)
        
                # clipping
                ratio = ratio.flatten(start_dim=1)
                surr1 = ratio * advantage
                surr2 = advantage * torch.clamp(ratio, 1-self.eps, 1+self.eps)

                value = self.value(states)
                value = (value - value.mean()) / (value.std () + 1e-8)
                
                # pi and value loss
                pi_loss = torch.mean(-torch.min(surr1, surr2))
                if self.algorithm == 'CCE-MAPPO':
                    KL = (ground_b*torch.exp((-torch.abs(mean - actions)/ground_b))+torch.abs(mean - actions))/b + torch.log(b/ground_b) - 1
                    pi_loss = pi_loss - torch.mean(self.kl_coef*KL)
                value_loss = torch.mean(F.mse_loss(value, td_target.detach()))

                total_loss = (pi_loss + 2*value_loss - new_policy.entropy() * self.entropy_coef)
                # total_loss = pi_loss + value_loss

                self.optimizer.zero_grad()
                total_loss.mean().backward()
                nn.utils.clip_grad_norm_(self.pi.parameters(), 10)
                self.optimizer.step()

                if epoch%5 == 0:
                    self.old_pi.load_state_dict(self.pi.state_dict())
                # self.old_value.load_state_dict(self.value.state_dict())
                start_index += 4

        with torch.no_grad():
            v = np.array([])
            for row in range(buffer_batchsize):
                state = transition[row]['states'][0]
                state = torch.stack(state, dim=0).flatten(start_dim=1)
                temp = self.value(state).squeeze(-1)
                temp = torch.mean(temp)
                v= np.append(v, temp.item())
            return v.mean()
        
