import pytorch_lightning as pl
import cv2
import matplotlib.pyplot as plt
import os
import torch, math
import yaml
import numpy as np
import torch.nn.functional as F
from algorithm.masac import MASAC
from algorithm.mappo import MAPPO
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser
from datasets import ArgoverseV2Dataset
from predictors import QCNet
from predictors.autoval import AntoQCNet
from predictors.environment import WorldModel
from transforms import TargetBuilder
from pathlib import Path
from utils.utils import get_transform_mat, get_auto_pred, add_new_agent, reward_function, save_gap, create_dir, save_reward
from torch_geometric.data import Batch
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="QCNet")
parser.add_argument("--root", type=str, default="/home/guanren/Multi-agent-competitive-environment/datasets")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--pin_memory", type=bool, default=True)
parser.add_argument("--persistent_workers", type=bool, default=True)
parser.add_argument("--accelerator", type=str, default="auto")
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--scenario", type=int, default=2)
parser.add_argument("--ckpt_path", default="checkpoints/epoch=10-step=274879.ckpt", type=str)
parser.add_argument("--RL_config", default="PPO_episode500_epoch20_beta1e-1_seed2023.yaml", type=str)
args = parser.parse_args()

with open("configs/"+args.RL_config+'.yaml', "r") as file:
        config = yaml.safe_load(file)
file.close()
pl.seed_everything(config['seed'], workers=True)
print(args.RL_config)
model = {
    "QCNet": AntoQCNet,
}[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path)
environment = {
    "QCNet": WorldModel,
}[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path)
val_dataset = {
    "argoverse_v2": ArgoverseV2Dataset,
}[model.dataset](
    root=args.root,
    split="val",
    transform=TargetBuilder(model.num_historical_steps, model.num_future_steps),
)

dataloader = DataLoader(
    val_dataset[[val_dataset.raw_file_names.index('0a8dd03b-02cf-4d7b-ae7f-c9e65ad3c900')]],
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=args.pin_memory,
    persistent_workers=args.persistent_workers,
)

trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices)

it = iter(dataloader)
data = next(it)

for param in model.encoder.parameters():
        param.requires_grad = False
for param in model.decoder.parameters():
        param.requires_grad = False
for param in environment.encoder.parameters():
        param.requires_grad = False
for param in environment.decoder.parameters():
        param.requires_grad = False

if isinstance(data, Batch):
    data['agent']['av_index'] += data['agent']['ptr'][:-1]

# new_input_data=data
v0_x = 1*math.cos(1.19)
v0_y = math.sqrt(1**2-v0_x**2)
new_input_data=add_new_agent(data,0.3, v0_x, v0_y, 1.19, 2665, -2410)
v0_x = 1*math.cos(-1.95)
v0_y = -math.sqrt(1**2-v0_x**2)
new_input_data=add_new_agent(new_input_data,0.3, v0_x, v0_y, -1.95, 2693, -2340)
v0_x = -1*math.cos(-0.33)
v0_y = math.sqrt(1**2-v0_x**2)
new_input_data=add_new_agent(new_input_data,-0.3, v0_x, v0_y, -0.33, 2725, -2386)
model_state_dict = torch.load('checkpoints/version_5/MASAC_episode500_epoch10_seed1234_task2.ckpt')

# next_version_path = create_dir(base_path = 'figures/')
cumulative_reward = [[] for _ in range(new_input_data['agent']['num_nodes'])]

choose_agent = []    
agent_index = -1
# agent_index = torch.nonzero(data['agent']['category']==3,as_tuple=False).item()
choose_agent.append(agent_index)
for i in range(config['agent_number']-1):
    choose_agent.append(data['agent']['num_nodes']+i)
offset=config['offset']

if 'MASAC' not in config['algorithm']:
    agents = [MAPPO(
                state_dim=model.num_modes*config['hidden_dim'],
                action_dim = model.output_dim*offset*6,
                config = config,
                device = model.device,
                offset = offset
        ) for _ in range(config['agent_number'])]
else:
    agents = [MASAC(
          state_dim=model.num_modes*config['hidden_dim'],
          action_dim = model.output_dim*offset*6,
          config = config,
          device = model.device,
          offset = offset
    ) for _ in range(config['agent_number'])]

for i in range(config['agent_number']):
    agents[i].critic_1.load_state_dict(model_state_dict[f'agent_{i}_critic'])
    agents[i].actor.load_state_dict(model_state_dict[f'agent_{i}_actor'])

v_array = np.array([])

with torch.no_grad():
    for episode in tqdm(range(config['episodes'])):
        
        scale=1/config['episodes']

        transition_list = [{
                    'states': [[]for _  in range(config['agent_number'])],
                    'actions': [[]for _  in range(config['agent_number'])],
                    'next_states': [[]for _  in range(config['agent_number'])],
                    'rewards': [[]for _  in range(config['agent_number'])],
                    'dones': []} for _ in range(config['buffer_batchsize'])]

        for batch in range(config['buffer_batchsize']):
            new_data=new_input_data.cuda().clone()

            pred = model(new_data)
            if model.output_head:
                traj_propose = torch.cat([pred['loc_propose_pos'][..., :model.output_dim],
                                        pred['loc_propose_head'],
                                        pred['scale_propose_pos'][..., :model.output_dim],
                                        pred['conc_propose_head']], dim=-1)
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :model.output_dim],
                                        pred['loc_refine_head'],
                                        pred['scale_refine_pos'][..., :model.output_dim],
                                        pred['conc_refine_head']], dim=-1)
            else:
                traj_propose = torch.cat([pred['loc_propose_pos'][..., :model.output_dim],
                                        pred['scale_propose_pos'][..., :model.output_dim]], dim=-1)
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :model.output_dim],
                                        pred['scale_refine_pos'][..., :model.output_dim]], dim=-1)

            auto_pred=pred
            origin,theta,rot_mat=get_transform_mat(new_data,model)
            new_true_trans_position_refine = torch.einsum(
                "bijk,bkn->bijn",
                pred["loc_refine_pos"][..., : model.output_dim],
                rot_mat.swapaxes(-1, -2),
            ) + origin[:, :2].unsqueeze(1).unsqueeze(1)

            init_origin,init_theta,init_rot_mat=get_transform_mat(new_data,model)
            pi = pred['pi']
            pi_eval = F.softmax(pi, dim=-1)

            state_temp_list = []
            global_state = environment.decoder(new_data, environment.encoder(new_data))
            for i in range(config['agent_number']):
                state_temp_list.append(global_state[choose_agent[i]])

            for timestep in range(0,model.num_future_steps,offset):

                true_trans_position_refine=new_true_trans_position_refine
                
                reg_mask_list = []
                for i in range(config['agent_number']):
                    reg_mask = new_data['agent']['predict_mask'][choose_agent[i], model.num_historical_steps:]
                    reg_mask_list.append(reg_mask)
                best_mode = pi_eval.argmax(dim=-1)
                max_value = -1
                max_index = -1
                for k in range(6):
                    x = auto_pred['loc_refine_pos'][agent_index, k, offset, 0]
                    y = auto_pred['loc_refine_pos'][agent_index, k, offset, 1]
                    value = torch.sqrt(x**2 + y**2)
                    if value > max_value:
                        max_value = value
                        max_index = k
                best_mode[agent_index] = max_index

                sample_action_list = []
                for i in range(config['agent_number']):
                    sample_action = agents[i].choose_action(state_temp_list[i], scale)
                    sample_action = sample_action.squeeze(0).reshape(-1,model.output_dim)
                    sample_action_list.append(sample_action)

                action_suggest_index_list = []
                action_suggest_index_list.append(max_index)
                for i in range(1, config['agent_number']):
                    l2_norm = (torch.norm(auto_pred['loc_refine_pos'][choose_agent[i],:,:offset, :2] -
                                        sample_action_list[i][:offset, :2].unsqueeze(0), p=2, dim=-1) * reg_mask_list[i][timestep:timestep+offset].unsqueeze(0)).sum(dim=-1)
                    action_suggest_index=l2_norm.argmin(dim=-1)
                    best_mode[choose_agent[i]] = action_suggest_index
                    action_suggest_index_list.append(action_suggest_index)

                action_list = []
                for i in range(config['agent_number']):
                    action = auto_pred['loc_refine_pos'][choose_agent[i],action_suggest_index_list[i],:offset, :2].flatten(start_dim = 1)
                    action_list.append(action)
                new_data, auto_pred, _, _, (new_true_trans_position_propose, new_true_trans_position_refine),(traj_propose, traj_refine) = get_auto_pred(
                    new_data, model, auto_pred["loc_refine_pos"][torch.arange(traj_propose.size(0)),best_mode], auto_pred["loc_refine_head"][torch.arange(traj_propose.size(0)),best_mode,:,0],offset,anchor=(init_origin,init_theta,init_rot_mat)
                )

                
                next_state_temp_list = []
                global_next_state = environment.decoder(new_data, environment.encoder(new_data))
                for i in range(config['agent_number']):
                    next_state_temp_list.append(global_next_state[choose_agent[i]])
                for i in range(config['agent_number']):
                    transition_list[batch]['states'][i].append(state_temp_list[i])
                    transition_list[batch]['actions'][i].append(action_list[i])
                for i in range(config['agent_number']):
                    reward = reward_function(new_input_data.clone(),new_data.clone(),model,choose_agent[i], config['agent_number'])
                    transition_list[batch]['rewards'][i].append(torch.tensor([reward]).cuda())
                        
                state_temp_list = next_state_temp_list
                        
                pi_eval = F.softmax(auto_pred['pi'], dim=-1)

        s = []
        a = []
        o = []
        for i in range(config['buffer_batchsize']):
            for j in range(config['agent_number']):
                  s+=transition_list[i]['states'][j]
            o += transition_list[i]['states'][-1]
            a += transition_list[i]['actions'][-1]
        states_critic_input = torch.stack(s, dim=0).reshape(-1,config['agent_number']*model.num_modes*config['hidden_dim']).type(torch.FloatTensor).to(model.device)
        actions_critic_input = torch.stack(a, dim=0).flatten(start_dim=1).reshape(-1, 2*5).type(torch.FloatTensor).to(model.device)
        observations = torch.stack(o, dim=0).reshape(-1,model.num_modes*config['hidden_dim']).type(torch.FloatTensor).to(model.device)

        q = agents[-1].critic_1(states_critic_input, actions_critic_input)
        _,_,actions, log_prob = agents[-1].actor(observations, scale)
        v = q - agents[-1].log_alpha.exp() * log_prob
        v = torch.mean(v.squeeze(-1)).item()
        print(v)
        v_array = np.append(v_array, v)

# save_reward('expert_'+args.RL_config+'_task'+str(args.task), next_version_path, cumulative_reward, config['agent_number'])
# save_gap('expert_'+args.RL_config+'_task'+str(args.task)+'_CCE-GAP_agent1', next_version_path, v_array.tolist())

