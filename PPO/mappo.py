import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import wrap_angle


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
            layer_init(nn.Linear(hidden_dim//2, action_dim))
        )
        self.hidden_dim = hidden_dim
    
    def forward(self, model, states, action, data, agent_index,offset):
        data['agent']['position'][agent_index,:50,:2]=states[:,:2]
        data['agent']['velocity'][agent_index,:50,:2]=states[:,2:4]
        data['agent']['heading'][agent_index,:50,:2]=states[:,-1]

        pred = model(data)
            
        reg_mask = data['agent']['predict_mask'][:-1, 50:]

        gt = torch.cat([data['agent']['target'][..., :2], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(pred['loc_refine_pos'][:-1,:,:, :2] -
                              gt[..., :2].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        best_mode = torch.cat([best_mode,torch.tensor([0]).cuda()])
        # best_mode = torch.randint(1,size=(data['agent']['num_nodes'],))
        loc_refine_pos = pred['loc_refine_pos'][torch.arange(pred['loc_refine_pos'].size(0)), best_mode,:2]
        loc_refine_head = pred["loc_refine_head"][torch.arange(pred['loc_refine_pos'].size(0)),best_mode,0]
        action_information = torch.cat([loc_refine_pos[agent_index,:offset,:2],loc_refine_head[agent_index,:offset,0].unsqueeze(-1)], dim=-1)
        
        mean = torch.tanh(self.f(action_information[agent_index]))
        var = F.softplus(self.f(action_information[agent_index])) + 1e-8
        return mean, var
    
class ValueNet(nn.Module):
    def __init__(self, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim*2)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim*2, hidden_dim*4)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim*4, hidden_dim*2)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim*2, hidden_dim//2)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim//2, 1)),
        )
        self.hidden_dim = hidden_dim

    def forward(self, encoder, states, agent_index):

        state_information = encoder(states)
        state_information = state_information['x_a'].reshape(-1, 50, self.hidden_dim)
        v = self.fc(state_information[agent_index])
        return v
    
class PPO:
    def __init__(self, 
                 batch_id,
                 encoder, 
                 decoder, 
                 data, 
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
        self.value = ValueNet(hidden_dim, state_dim).to(device)
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

    def choose_action(self, model, state, data, offset):
        with torch.no_grad():
            mean, var = self.old_pi(model, state, data, offset)
            dis = Normal(mean, var)
            a = dis.sample()
        return a

    
    def update(self, transition, agent_index):
        states = torch.cat(transition[agent_index]['states'], dim=0).to(self.device)
        actions = torch.cat(transition[agent_index]['actions'],dim=0).to(self.device)  
        next_states = torch.cat(transition[agent_index]['next_states'],dim=0).to(self.device)  
        dones = torch.tensor(transition[agent_index]['dones'], dtype=torch.float).view(-1,1).to(self.device)  
        rewards = torch.tensor(transition[agent_index]['rewards'], dtype=torch.float).view(-1,1).to(self.device)

        print(states.shape)

        for epoch in range(self.epochs):
 
            # td_error
            next_state_value = self.value(self.encoder, self.decoder, next_states, self.data, self.agent_index, 1, self.offset)
            td_target = rewards + self.gamma * next_state_value.view(-1,self.offset) * (1-dones)
            td_value = self.value(self.encoder, self.decoder, states, self.data, self.agent_index, 0, self.offset)
            td_delta = td_value.view(-1,self.offset) - td_target

            td_target = td_target.view(-1,1)
            td_delta = td_delta.view(-1,1)
    
            # calculate GAE
            advantage = 0
            advantage_list = []
            td_delta = td_delta.cpu().detach().numpy()
            for delta in td_delta[::-1]:
                with torch.no_grad():
                    advantage = self.gamma * self.lamda * advantage + delta
                advantage_list.append(advantage)
            advantage_list.reverse()
            advantage = torch.from_numpy(np.array(advantage_list, dtype=np.float32)).to(self.device)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    
            # get ratio
            old_dis = self.old_pi(self.encoder, self.decoder, states, self.data, self.agent_index)
            old_log_probs = old_dis.log_prob(actions)
            dis = self.pi(self.encoder, self.decoder, states, self.data, self.agent_index)
            log_probs = dis.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
    
            # clipping
            surr1 = advantage * ratio
            surr2 = advantage * torch.clamp(ratio, 1-self.eps, 1+self.eps)
            
            # pi and value loss
            pi_loss = torch.mean(-torch.min(surr1, surr2))
            value_loss = torch.mean(F.mse_loss(td_value, td_target.detach()))

            total_loss = (pi_loss + value_loss - dis.entropy() * self.entropy_coef)
            
            self.optimizer.zero_grad()
            total_loss.mean().requires_grad_(True).backward()
            self.optimizer.step()

        self.old_pi.load_state_dict(self.pi.state_dict())
