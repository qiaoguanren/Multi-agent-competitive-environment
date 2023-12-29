import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from tqdm import tqdm


def layer_init(layer, std=0, bias_const=0.0):
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
            layer_init(nn.Linear(hidden_dim//2, action_dim))
        )
        self.hidden_dim = hidden_dim
    
    def forward(self, encoder, decoder, data, agent_index, offset):

        states = encoder(data)
        pred = decoder(data, states)
            
        reg_mask = data['agent']['predict_mask'][:-1, 50:]

        gt = torch.cat([data['agent']['target'][..., :2], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(pred['loc_refine_pos'][:-1,:,:, :2] -
                              gt[..., :2].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        best_mode = torch.cat([best_mode,torch.tensor([0]).cuda()])
        loc_refine_pos = pred['loc_refine_pos'][torch.arange(pred['loc_refine_pos'].size(0)), best_mode,:,:2]
        loc_refine_head = pred["loc_refine_head"][torch.arange(pred['loc_refine_pos'].size(0)),best_mode,:,0]
        action_information = torch.cat([loc_refine_pos[agent_index,:,:2],loc_refine_head[agent_index,:].unsqueeze(-1)], dim=-1)
        
        mean = self.f(action_information)
        mean = torch.cat([mean[:,:2],(torch.tanh(mean[:,-1].clone().detach().requires_grad_(True))*torch.tensor([math.pi]).cuda()).unsqueeze(-1)],dim=-1)
        var = F.softplus(self.f(action_information)) + 1e-8
        return mean, var
    
class ValueNet(nn.Module):
    def __init__(self, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim*2)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim*2, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim//2)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim//2, hidden_dim//4)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim//4, 1)),
        )
        self.hidden_dim = hidden_dim

    def forward(self, states, agent_index):

        v = self.fc(states)
        return v
    
class PPO:
    def __init__(self, 
                 batch_id,
                 encoder, 
                 decoder,  
                 agent_index: int, 
                 hidden_dim: int,
                 action_dim: int,
                 state_dim: int,
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
        self.value = ValueNet(hidden_dim).to(device)
        self.batch_id = batch_id 
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

    def choose_action(self, encoder, decoder, data, agent_index, offset):
        with torch.no_grad():
            mean, var = self.old_pi(encoder, decoder, data, agent_index, offset)
            dis = Normal(mean, var)
            a = dis.sample()
        return a

    
    def update(self, transition, data, agent_index):
        states = transition[agent_index]['states'][0]['x_a'][agent_index,:,:]
        for i in range(1,len(transition[agent_index]['states'])):
            states = torch.cat([states,transition[agent_index]['states'][i]['x_a'][agent_index,:,:]],dim=0)
        actions = torch.cat(transition[agent_index]['actions'],dim=0).to(self.device)  
        next_states = transition[agent_index]['next_states'][0]['x_a'][agent_index,:,:]
        for i in range(1,len(transition[agent_index]['next_states'])):
            next_states = torch.cat([next_states,transition[agent_index]['next_states'][i]['x_a'][agent_index,:,:]],dim=0)  
        dones = torch.tensor(transition[agent_index]['dones'], dtype=torch.float).view(-1,1).to(self.device)  
        rewards = torch.tensor(transition[agent_index]['rewards'], dtype=torch.float).view(-1,1).to(self.device)

        for epoch in tqdm(range(self.epochs)):
 
            with torch.no_grad():
                # td_error
                next_state_value = self.value(next_states, self.agent_index)
                td_target = rewards + self.gamma * next_state_value.view(int(60/self.offset),-1) * (1-dones)
                td_value = self.value(states, self.agent_index)
                td_delta = td_value.view(int(60/self.offset),-1) - td_target

                td_delta = td_delta.view(60,-1)
        
                # calculate GAE
                advantage = 0
                advantage_list = []
                td_delta = td_delta.cpu().detach().numpy()
                for delta in td_delta[::-1]:
                    advantage = self.gamma * self.lamda * advantage + delta
                    advantage_list.append(advantage)
                advantage_list.reverse()
                advantage = torch.from_numpy(np.array(advantage_list, dtype=np.float32)).to(self.device)
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
                # get ratio
                mean, var = self.old_pi(self.encoder, self.decoder, data, self.agent_index, self.offset)
                old_policy = Normal(mean, var)
                old_log_probs = old_policy.log_prob(actions)

            mean, var = self.pi(self.encoder, self.decoder, data, self.agent_index, self.offset)
            new_policy = Normal(mean, var)
            log_probs = new_policy.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
    
            # clipping
            advantage = advantage.unsqueeze(1)
            advantage = advantage.repeat(1, 3, 1)
            ratio = ratio.unsqueeze(-1)
            ratio = ratio.repeat(1, 1, 10)
            surr1 = advantage * ratio
            surr2 = advantage * torch.clamp(ratio, 1-self.eps, 1+self.eps)
            
            # pi and value loss
            pi_loss = torch.mean(-torch.min(surr1, surr2))
            value_loss = torch.mean(F.mse_loss(self.value(states, self.agent_index), td_target.reshape(-1,1).detach()))

            total_loss = (pi_loss + value_loss - new_policy.entropy() * self.entropy_coef)
            
            self.optimizer.zero_grad()
            total_loss.mean().requires_grad_(True).backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                self.old_pi.load_state_dict(self.pi.state_dict())
