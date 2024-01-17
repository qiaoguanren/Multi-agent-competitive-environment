import torch
import math, random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from tqdm import tqdm
from visualization.vis import vis_entropy


def layer_init(layer, std=np.sqrt(2), bias_const=0.1):
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
            layer_init(nn.Linear(hidden_dim, action_dim), bias_const=1.0),
        )
    
    def forward(self, state, scale):

        mean = self.f(state) + scale
        var = F.sigmoid(self.f(state))*1e-1 + 1e-8
        return mean, var
    
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(hidden_dim, 1)),

        )# two-layer MLP

    def forward(self, states):

        v = self.fc(states)
        return v
    
class PPO:
    def __init__(self, 
                 state_dim: int, 
                 agent_index: int,
                 hidden_dim: int,
                 action_dim: int,
                 batchsize: int,
                 actor_learning_rate: float,
                 critic_learning_rate: float,
                 lamda: float,
                 eps: float, 
                 gamma: float, 
                 device, 
                 agent_num: int, 
                 offset: int, 
                 entropy_coef: float,
                 epochs: int):
        self.pi = Policy(state_dim, hidden_dim, action_dim).to(device)
        self.old_pi = Policy(state_dim, hidden_dim, action_dim).to(device)
        self.value = ValueNet(state_dim, hidden_dim).to(device)
        # self.old_value = ValueNet(state_dim, hidden_dim).to(device)
        self.batchsize = batchsize
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.lamda = lamda #discount factor
        self.eps = eps #clipping parameter
        self.gamma = gamma # the factor of caculating GAE
        self.device = device
        self.agent_num = agent_num
        self.agent_index = agent_index
        self.offset = offset
        self.entropy_coef = entropy_coef
        self.epochs = epochs

        self.optimizer = torch.optim.AdamW([
                        {'params': self.pi.parameters(), 'lr': self.actor_lr},
                        {'params': self.value.parameters(), 'lr': self.critic_lr}
                    ])
        
    def sample(self, transition, sample_batchsize, buffer_batchsize, offset):

        s = []
        a = []
        r = []
        s_next = []
        done = []

        with torch.no_grad():
            for _ in range(sample_batchsize):
            # for _ in range(offset):
                row = random.randint(0,buffer_batchsize-1)
                col = random.randint(0,int(60/offset)-1)
                s.append(transition[row]['states'][col])
                a.append(transition[row]['actions'][col])
                r.append(transition[row]['rewards'][col])
                s_next.append(transition[row]['next_states'][col])
                done.append(transition[row]['dones'][col])
                # s += transition[row]['states']
                # a += transition[row]['actions']
                # r += transition[row]['rewards']
                # s_next += transition[row]['next_states']
                # done += transition[row]['dones']

        return s, a, r, s_next, done

    def choose_action(self, state, scale):
        with torch.no_grad():
            state = torch.flatten(state,start_dim=0).unsqueeze(0)
            mean, var = self.old_pi(state, scale)
            action = Normal(mean, var)
            action = action.sample()
            # samples = Normal(mean, var).sample((10,))
            # action = samples.max(dim=0)[0]
        return action

    
    def update(self, transition, buffer_batchsize, episode, version_path, scale):

        entropy_list = []
        
        for epoch in tqdm(range(self.epochs)):

            states,  actions, rewards, next_states, dones= self.sample(transition, self.batchsize, buffer_batchsize, self.offset)
            states = torch.stack(states, dim=0).flatten(start_dim=1)
            next_states = torch.stack(next_states, dim=0).flatten(start_dim=1)
            rewards = torch.stack(rewards, dim=0).view(-1,1)
            dones = torch.stack(dones, dim=0).view(-1,1)
            actions = torch.stack(actions, dim=0).flatten(start_dim=1)
                
            # td_error
            with torch.no_grad():
                next_state_value = self.value(next_states)
                td_target = rewards + self.gamma * next_state_value * (1-dones)
                td_value = self.value(states)
                td_delta = td_value - td_value
            
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
                mean, var = self.old_pi(states, scale)
                old_policy = Normal(mean, var)
                old_log_probs = old_policy.log_prob(actions)

            mean, var = self.pi(states, scale)
            new_policy = Normal(mean, var)
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

            total_loss = (pi_loss + 0.5*value_loss - new_policy.entropy() * self.entropy_coef)
            # total_loss = pi_loss + value_loss
            entropy_list.append(new_policy.entropy().mean().item())

            self.optimizer.zero_grad()
            total_loss.mean().backward()
            nn.utils.clip_grad_norm_(self.pi.parameters(), 0.5)
            self.optimizer.step()

            if epoch%10 == 0:
                self.old_pi.load_state_dict(self.pi.state_dict())
            # self.old_value.load_state_dict(self.value.state_dict())

        if (episode+1)%100==0:
            if version_path:
                vis_entropy(entropy_list, episode, version_path)

        
