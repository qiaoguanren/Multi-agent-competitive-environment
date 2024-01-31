import pytorch_lightning as pl
import cv2
import matplotlib.pyplot as plt
import os
import torch, math
import yaml
import numpy as np
import torch.nn.functional as F
from algorithm.ppo import PPO
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser
from datasets import ArgoverseV2Dataset
from predictors import QCNet
from predictors.autoval import AntoQCNet
from predictors.environment import WorldModel
from transforms import TargetBuilder
from pathlib import Path
from visualization.vis import vis_reward
from utils.geometry import wrap_angle
from utils.utils import get_transform_mat, get_auto_pred, add_new_agent, reward_function, sample_from_pdf, create_dir, save_reward
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
parser.add_argument("--task", type=int, default=1)
parser.add_argument("--ckpt_path", default="checkpoints/epoch=10-step=274879.ckpt", type=str)
parser.add_argument("--RL_config", default="PPO_episode500_epoch20_beta1e-1_seed2023", type=str)
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
next_version_path = create_dir(base_path = 'figures/')
cumulative_reward = []

offset=config['offset']
agent = PPO(
            state_dim=model.num_modes*config['hidden_dim'],
            action_dim = model.output_dim*offset*6,
            config = config,
            device = model.device,
            offset = offset
)

for episode in tqdm(range(config['episodes'])):
    scale=1/config['episodes']
    transition_list = [{
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []} for _ in range(config['buffer_batchsize'])]
        
    agent_index = torch.nonzero(data['agent']['category']==3,as_tuple=False).item()

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

        state = environment.decoder(new_data, environment.encoder(new_data))[agent_index]

        for timestep in range(0,model.num_future_steps,offset):

            true_trans_position_refine=new_true_trans_position_refine
            reg_mask = new_data['agent']['predict_mask'][agent_index, model.num_historical_steps:]
            
            sample_action = agent.choose_action(state, scale)
            sample_action = sample_action.squeeze(0).reshape(-1,model.output_dim)

            best_mode = pi_eval.argmax(dim=-1)
            l2_norm = (torch.norm(auto_pred['loc_refine_pos'][agent_index,:,:offset, :2]-
                                sample_action[:offset, :2].unsqueeze(0), p=2, dim=-1) * reg_mask[timestep:timestep+offset].unsqueeze(0)).sum(dim=-1)
            action_suggest_index=l2_norm.argmin(dim=-1)
            best_mode[agent_index] = action_suggest_index

            action = auto_pred['loc_refine_pos'][agent_index,action_suggest_index,:offset, :2].flatten(start_dim = 1)
            new_data, auto_pred, _, _, (new_true_trans_position_propose, new_true_trans_position_refine),(traj_propose, traj_refine) = get_auto_pred(
                new_data, model, auto_pred["loc_refine_pos"][torch.arange(traj_propose.size(0)),best_mode], auto_pred["loc_refine_head"][torch.arange(traj_propose.size(0)),best_mode,:,0],offset,anchor=(init_origin,init_theta,init_rot_mat)
            )

            with torch.no_grad():
                next_state = environment.decoder(new_data, environment.encoder(new_data))[agent_index]
                transition_list[batch]['states'].append(state)
                transition_list[batch]['actions'].append(action)
                transition_list[batch]['next_states'].append(next_state)
                if timestep == model.num_future_steps - offset:
                    transition_list[batch]['dones'].append(torch.tensor(1).cuda())
                else:
                    transition_list[batch]['dones'].append(torch.tensor(0).cuda())
                reward = reward_function(new_input_data.clone(),new_data.clone(),model,agent_index, config['agent_number'])
                transition_list[batch]['rewards'].append(torch.tensor([reward]).cuda())
                     
            state = next_state
            pi_eval = F.softmax(auto_pred['pi'], dim=-1)
            
    agent.update(transition_list, config['buffer_batchsize'], episode, next_version_path, scale)

    discounted_return = 0
    undiscounted_return = 0

    for t in reversed(range(0, int(model.num_future_steps/offset))):
        _sum = 0
        for i in range(config['buffer_batchsize']):
             _sum+=float(transition_list[i]['rewards'][t])
        mean_reward = _sum/config['buffer_batchsize']
        discounted_return = (config['gamma'] * discounted_return) + mean_reward
        undiscounted_return += mean_reward

    print(discounted_return, undiscounted_return)

    cumulative_reward.append(discounted_return)

save_reward(args.RL_config+'_task'+str(args.task), next_version_path, cumulative_reward, config['agent_number'])
# vis_reward(new_data,cumulative_reward,agent_index+1,config['episodes'],next_version_path)

# pi_state_dict = agent.pi.state_dict()
# old_pi_state_dict = agent.old_pi.state_dict()
# # old_value_state_dict = agent.old_value.state_dict()
# value_state_dict = agent.value.state_dict()
# model_state_dict = {
#     'pi': pi_state_dict,
#     # 'old_value': old_value_state_dict,
#     'value': value_state_dict
# }

# next_version_path = create_dir(base_path = 'checkpoints/')
# torch.save(model_state_dict, next_version_path+args.RL_config+'.ckpt')

