import torch
import math, random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Laplace
import numpy as np
from tqdm import tqdm
from utils.normalizing_flow import MADE, BatchNormFlow, Reverse, FlowSequential
from visualization.vis import vis_entropy


def layer_init(layer, std=0, bias_const=0.0):
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
            layer_init(nn.Linear(hidden_dim, action_dim)),
        )
    
    def forward(self, state, scale):

        noise = Normal(scale, 0.1)
        mean = self.f(state) + noise.sample()
        mean = torch.cumsum(mean.reshape(-1,6,5,2), dim=-2)
        mean = mean[:, -1, :, :]
        mean = mean.flatten(start_dim = 1)

        b = self.f(state)
        b = torch.cumsum(F.elu_(b.reshape(-1,6,5,2),alpha = 1.0) + 1.0, dim=-2) + 0.1
        b = b[:, -1, :, :]
        b = b.flatten(start_dim = 1)
        return mean, b

    
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, agent_number):
        super(ValueNet, self).__init__()

        self.f = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(hidden_dim, 1)),
        )

        self.linear = nn.Linear(state_dim, state_dim//agent_number)

    def forward(self, states, agent_number):

        if agent_number > 1:
        
            states = states.flatten(start_dim = 0)
            states = self.linear(states)
            states = states.reshape(-1, 1)

        v = self.f(states)
        return v
    
class PPO:
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 config,
                 device, 
                 offset: int):
        self.hidden_dim =config['hidden_dim']
        # self.old_value = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_lr = config['actor_learning_rate']
        self.critic_lr = config['critic_learning_rate']
        self.density_lr = config['density_learning_rate']
        self.beta_coef = config['beta_coef']
        self.lamda = config['lamda'] #discount factor
        self.eps = config['eps'] #clipping parameter
        self.gamma = config['gamma'] # the factor of caculating GAE
        self.device = device
        self.offset = offset
        self.entropy_coef = config['entropy_coef']
        self.kl_coef = config['kl_coef']
        self.epochs = config['epochs']
        self.agent_number = config['agent_number']
        self.algorithm = config['algorithm']

        self.pi = Policy(state_dim, self.hidden_dim, action_dim).to(device)
        self.old_pi = Policy(state_dim, self.hidden_dim, action_dim).to(device)
        self.value = ValueNet(state_dim, self.hidden_dim, self.agent_number).to(device)

        self.optimizer = torch.optim.AdamW([
                        {'params': self.pi.parameters(), 'lr': self.actor_lr, 'eps': 1e-5},
                        {'params': self.value.parameters(), 'lr': self.critic_lr, 'eps': 1e-5}
                    ])
        self._init_density_model(state_dim, self.hidden_dim)
        
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
        
    def sample(self, transition, start_index):

        s = []
        a = []
        r = []
        s_next = []
        done = []

        with torch.no_grad():
            for row in range(start_index, start_index+4):
                s += transition[row]['states']
                a += transition[row]['actions']
                r += transition[row]['rewards']
                s_next += transition[row]['next_states']
                done += transition[row]['dones']

        return s, a, r, s_next, done

    def choose_action(self, state, scale):
        with torch.no_grad():
            state = torch.flatten(state,start_dim=0).unsqueeze(0)
            mean, var = self.old_pi(state, scale)
            action = Laplace(mean, var)
            action = action.sample()
        return action

    
    def update(self, transition, buffer_batchsize, episode, version_path, scale):
        
        for epoch in tqdm(range(self.epochs)):

            for _ in range(8):
            
                start_index = 0

                states,  actions, rewards, next_states, dones= self.sample(transition, start_index)
                states = torch.stack(states, dim=0).flatten(start_dim=1)
                next_states = torch.stack(next_states, dim=0).flatten(start_dim=1)
                rewards = torch.stack(rewards, dim=0).view(-1,1)
                dones = torch.stack(dones, dim=0).view(-1,1)
                actions = torch.stack(actions, dim=0).flatten(start_dim=1)
                rewards = rewards / (rewards.std() + 1e-5)

                # td_error
                with torch.no_grad():
                    next_state_value = self.value(next_states, self.agent_number)
                    next_state_value = (next_state_value - next_state_value.mean()) / (next_state_value.std() + 1e-5)
                    td_target = rewards + self.gamma * next_state_value * (1-dones)
                    # _, log_prob_game = self.density_model.log_probs(inputs=next_states,
                    #                                                 cond_inputs=None)
                    # log_prob_game = F.sigmoid((log_prob_game - log_prob_game.mean()) / log_prob_game.std())
                    # beta_t = self.beta_coef / (log_prob_game)
                    
                    # td_target = td_target + beta_t
                    td_value = self.value(states, self.agent_number)
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
                    mean, b = self.old_pi(states, scale)
                    old_policy = Laplace(mean, b)
                    old_log_probs = old_policy.log_prob(actions)

                mean, b = self.pi(states, scale)
                new_policy = Laplace(mean, b)
                log_probs = new_policy.log_prob(actions)
                ratio = torch.exp(log_probs - old_log_probs)
        
                # clipping
                ratio = ratio.flatten(start_dim=1)
                surr1 = ratio * advantage
                surr2 = advantage * torch.clamp(ratio, 1-self.eps, 1+self.eps)

                value = self.value(states, self.agent_number)
                value = (value - value.mean()
                        ) / (value.std() + 1e-8)
                
                # pi and value loss
                pi_loss = torch.mean(-torch.min(surr1, surr2))
                value_loss = torch.mean(F.mse_loss(value, td_target.detach()))

                total_loss = (pi_loss + 2*value_loss - new_policy.entropy() * self.entropy_coef)
                # total_loss = pi_loss + value_loss
                # entropy_list.append(new_policy.entropy().mean().item())

                self.optimizer.zero_grad()
                total_loss.mean().backward()
                nn.utils.clip_grad_norm_(self.pi.parameters(), 10)
                self.optimizer.step()

                if epoch%5 == 0:
                    self.old_pi.load_state_dict(self.pi.state_dict())
                # self.old_value.load_state_dict(self.value.state_dict())

                start_index += 4

            # if (episode+1)%100==0:
            #     if version_path:
            #         vis_entropy(entropy_list, episode, version_path)

        
