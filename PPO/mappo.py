import torch
import math, random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from tqdm import tqdm


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

class Policy(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super(Policy, self).__init__()
        self.f = nn.Sequential(
            layer_init(nn.Linear(action_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim//2)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim//2, action_dim),std=0.01)
        )
        self.hidden_dim = hidden_dim
    
    def forward(self, state, decoder, datas, agent_index, offset):

        mu = []
        sigma = []

        for i in range(len(datas)):

            data = datas[i]

            pred = decoder(data, state[i])
            
            reg_mask = data['agent']['predict_mask'][:-1, 50:]

            gt = torch.cat([data['agent']['target'][..., :2], data['agent']['target'][..., -1:]], dim=-1)
            l2_norm = (torch.norm(pred['loc_refine_pos'][:-1,:,:, :2] -
                                gt[..., :2].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
            best_mode = l2_norm.argmin(dim=-1)
            best_mode = torch.cat([best_mode,torch.tensor([0]).cuda()])
            loc_refine_pos = pred['loc_refine_pos'][torch.arange(pred['loc_refine_pos'].size(0)), best_mode,:,:2]
            loc_refine_head = pred["loc_refine_head"][torch.arange(pred['loc_refine_pos'].size(0)),best_mode,:,0]
            action_information = torch.cat([loc_refine_pos[agent_index,:offset,:2],loc_refine_head[agent_index,:offset].unsqueeze(-1)], dim=-1)
            
            mean = self.f(action_information)
            mean = torch.cat([mean[:,:2],(torch.tanh(mean[:,-1].clone().detach().requires_grad_(True))*torch.tensor([math.pi]).cuda()).unsqueeze(-1)],dim=-1)
            var = F.softplus(self.f(action_information)) + 1e-8
            mu.append(mean)
            sigma.append(var)
        return mu, sigma
    
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim*2)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim*2, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim//2)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim//2, hidden_dim//4)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim//4, 1), std=1.0),
        )
        self.hidden_dim = hidden_dim

    def forward(self, states):

        v = self.fc(states)
        return v
    
class PPO:
    def __init__(self, 
                 state_dim,
                 encoder, 
                 decoder,  
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
        self.pi = Policy(hidden_dim, action_dim).to(device)
        self.old_pi = Policy(hidden_dim, action_dim).to(device)
        self.value = ValueNet(state_dim, hidden_dim).to(device)
        self.batchsize = batchsize
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.lamda = lamda #discount factor
        self.eps = eps #clipping parameter
        self.gamma = gamma # the factor of caculating GAE
        self.device = device
        self.agent_num = agent_num
        self.encoder = encoder
        self.decoder = decoder
        self.agent_index = agent_index
        self.offset = offset
        self.entropy_coef = entropy_coef
        self.epochs = epochs

        self.optimizer = torch.optim.Adam([
                        {'params': self.pi.parameters(), 'lr': self.actor_lr},
                        {'params': self.value.parameters(), 'lr': self.critic_lr}
                    ])
        
    def sample(self, transition, sample_batchsize, buffer_batchsize, offset):

        data = []
        s = []
        a = []
        r = []
        s_next = []
        done = []

        for batch in range(sample_batchsize):

            row = random.randint(0,buffer_batchsize-1)
            col = random.randint(0,int(60/offset)-1)
            data.append(transition[row]['datas'][col])
            s.append(transition[row]['states'][col])
            a.append(transition[row]['actions'][col])
            r.append(transition[row]['rewards'][col])
            s_next.append(transition[row]['next_states'][col])
            done.append(transition[row]['dones'][col])

        return data, s, a, r, s_next, done

    def choose_action(self, state, decoder, data, agent_index, offset):
        with torch.no_grad():
            mu, sigma = self.old_pi(state, decoder, data, agent_index, offset)
            mean = mu[0]
            var = sigma[0]
            dis = Normal(mean, var)
            a = dis.sample()
        return a

    
    def update(self, transition, buffer_batchsize):
        
        for epoch in tqdm(range(self.epochs)):
            data, states,  actions, rewards, next_states, dones = self.sample(transition, self.batchsize, buffer_batchsize, self.offset)

            x_a = [state['x_a'][self.agent_index] for state in states]
            critic_states = torch.stack(x_a, dim=0).flatten(start_dim=1)
            n_x_a = [state['x_a'][self.agent_index] for state in next_states]
            critic_next_states = torch.stack(n_x_a, dim=0).flatten(start_dim=1)
            rewards = torch.stack(rewards, dim=0).view(-1,1)
            dones = torch.stack(dones, dim=0).view(-1,1)
            actions = torch.stack(actions, dim=0)

            with torch.no_grad():
                # td_error
                next_state_value = self.value(critic_next_states)
                td_target = rewards + self.gamma * next_state_value * (1-dones)
                td_value = self.value(critic_states)
                td_delta = td_target - td_value
        
                # calculate GAE
                advantage = 0
                advantage_list = []
                td_delta = td_delta.cpu().detach().numpy()
                for delta in td_delta[::-1]:
                    advantage = self.gamma * self.lamda * advantage + delta[0]
                    advantage_list.append(advantage)
                advantage_list.reverse()
                advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device).reshape(-1,1)
                #advantage_normalization
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
                # get ratio
                mu, sigma = self.old_pi(states, self.decoder, data, self.agent_index, self.offset)
                mean = torch.stack(mu, dim=0)
                var = torch.stack(sigma, dim=0)
                old_policy = Normal(mean, var)
                old_log_probs = old_policy.log_prob(actions)

            mu, sigma = self.pi(states, self.decoder, data, self.agent_index, self.offset)
            mean = torch.stack(mu, dim=0)
            var = torch.stack(sigma, dim=0)
            new_policy = Normal(mean, var)
            log_probs = new_policy.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
    
            # clipping
            ratio = ratio.flatten(start_dim=1)
            surr1 = ratio * advantage
            surr2 = advantage * torch.clamp(ratio, 1-self.eps, 1+self.eps)
            
            # pi and value loss
            pi_loss = torch.mean(-torch.min(surr1, surr2))
            value_loss = torch.mean(F.mse_loss(self.value(critic_states), td_target.detach()))

            total_loss = (pi_loss + value_loss - new_policy.entropy() * self.entropy_coef)
            # total_loss = pi_loss + value_loss
            
            self.optimizer.zero_grad()
            total_loss.mean().requires_grad_(True).backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                self.old_pi.load_state_dict(self.pi.state_dict())
