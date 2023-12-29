import pytorch_lightning as pl
import cv2
import matplotlib.pyplot as plt
import io
import torch, math
import yaml
import numpy as np
from PPO.mappo import PPO
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser
from datasets import ArgoverseV2Dataset
from predictors import QCNet
from predictors.autoval import AntoQCNet
from transforms import TargetBuilder
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
    TrackCategory,
)
from av2.map.map_api import ArgoverseStaticMap
from pathlib import Path
from visualization.vis import vis_reward, generate_video
from utils.utils import get_transform_mat, get_auto_pred, add_new_agent, reward_function
from torch_geometric.data import Batch

pl.seed_everything(2023, workers=True)

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="QCNet")
parser.add_argument("--root", type=str, default="/home/guanren/Multi-agent-competitive-environment/datasets")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--pin_memory", type=bool, default=True)
parser.add_argument("--persistent_workers", type=bool, default=True)
parser.add_argument("--accelerator", type=str, default="auto")
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--ckpt_path", default="checkpoints/epoch=10-step=274879.ckpt", type=str)
args = parser.parse_args("")

model = {
    "QCNet": AntoQCNet,
}[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path)
val_dataset = {
    "argoverse_v2": ArgoverseV2Dataset,
}[model.dataset](
    root=args.root,
    split="val",
    transform=TargetBuilder(model.num_historical_steps, model.num_future_steps),
)

dataloader = DataLoader(
    val_dataset[[val_dataset.raw_file_names.index('0a0ef009-9d44-4399-99e6-50004d345f34')]],
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

if isinstance(data, Batch):
    data['agent']['av_index'] += data['agent']['ptr'][:-1]

new_input_data=add_new_agent(data)
init_origin,init_theta,init_rot_mat=get_transform_mat(new_input_data,model)

episodes = 300
cumulative_reward = [{'return': []} for _ in range(new_input_data['agent']['num_nodes'])]

with open("configs/PPO.yaml", "r") as file:
        config = yaml.safe_load(file)
file.close()

for episode in range(episodes):
    offset=5
    new_data=new_input_data.cuda().clone()
    pred = model(new_data)
    traj_propose = torch.cat([pred['loc_propose_pos'][..., :model.output_dim],
                            pred['scale_propose_pos'][..., :model.output_dim]], dim=-1)
    traj_refine = torch.cat([pred['loc_refine_pos'][..., :model.output_dim],
                            pred['scale_refine_pos'][..., :model.output_dim]], dim=-1)
    final_traj_propose=torch.zeros_like(traj_propose)
    final_traj_refine=torch.zeros_like(traj_refine)
    true_traj_propose=torch.zeros_like(traj_propose)
    true_traj_refine=torch.zeros_like(traj_refine)
    auto_pred=pred

    transition_list = [{'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': []} for _ in range(new_data['agent']['num_nodes'])]

    for timestep in range(0,model.num_future_steps,offset):
        reg_mask = new_data['agent']['predict_mask'][:, model.num_historical_steps:]
        cls_mask = new_data['agent']['predict_mask'][:, -1]
        # gt = torch.cat([data['agent']['target'][...,timestep:timestep+offset, :model.output_dim], data['agent']['target'][...,timestep:timestep+offset, -1:]], dim=-1)
        # l2_norm = (torch.norm(traj_propose[...,:offset, :model.output_dim] -
        #                     gt[..., :model.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask[...,:offset].unsqueeze(1)).sum(dim=-1)
        # best_mode = l2_norm.argmin(dim=-1)
        best_mode = torch.randint(1,size=(new_data['agent']['num_nodes'],))
            
        agent_index = torch.nonzero(new_data['agent']['category']==3,as_tuple=False).item()
        state = new_data['agent']['position'][agent_index, model.num_historical_steps - offset : model.num_historical_steps, :model.output_dim]
        state = torch.cat([state, new_data['agent']['velocity'][agent_index, model.num_historical_steps - offset : model.num_historical_steps, :model.output_dim]], dim = -1)
        next_state = new_data['agent']['position'][agent_index, model.num_historical_steps : model.num_historical_steps + offset, :model.output_dim]
        next_state = torch.cat([next_state, new_data['agent']['velocity'][agent_index, model.num_historical_steps : model.num_historical_steps + offset, :model.output_dim]], dim = -1)
            
        action = new_data['agent']['heading'][agent_index, model.num_historical_steps - offset : model.num_historical_steps]
        acceleration = (new_data['agent']['velocity'][agent_index, model.num_historical_steps : model.num_historical_steps + offset, :model.output_dim] - new_data['agent']['velocity'][agent_index, model.num_historical_steps - offset: model.num_historical_steps, :model.output_dim])/(0.1*offset)
        action = torch.cat([action.unsqueeze(-1), acceleration],dim=-1)
        # a = (a-torch.min(a))/(torch.max(a)-torch.min(a))
        # a = 2 * a - 1

        transition_list[agent_index]['states'].append(state)
        transition_list[agent_index]['actions'].append(action)
        transition_list[agent_index]['next_states'].append(next_state)

        transition_list[agent_index]['dones'].append(False)
        reward = reward_function(new_data, model, agent_index)
        transition_list[agent_index]['rewards'].append(reward)
                

        new_data, auto_pred, _, _, (true_trans_position_propose, true_trans_position_refine),(traj_propose, traj_refine) = get_auto_pred(
            new_data, model, auto_pred["loc_refine_pos"][torch.arange(traj_propose.size(0)),best_mode], auto_pred["loc_refine_head"][torch.arange(traj_propose.size(0)),best_mode,:,0], offset,anchor=(init_origin,init_theta,init_rot_mat)
        )
        true_traj_refine[:,:,timestep:timestep+offset,:2]=new_data["agent"]["position"][...,model.num_historical_steps - offset : model.num_historical_steps,:2].unsqueeze(1)
        pi = auto_pred['pi']


    agent_index = torch.nonzero(new_data['agent']['category']==3,as_tuple=False).item()
            
    agent = PPO(
            batch_id=0,
            encoder = model.encoder,
            decoder = model.decoder,
            data = new_data.clone().cuda(),
            agent_index = agent_index,
            hidden_dim = config['hidden_dim'],
            action_dim = model.output_dim+1,
            state_dim = model.output_dim*2,
            actor_learning_rate = config['actor_learning_rate'],
            critic_learning_rate = config['critic_learning_rate'],
            lamda = config['lamda'],
            eps = config['eps'],
            gamma = config['gamma'],
            device = model.device,
            agent_num = new_data['agent']['num_nodes'],
            offset = offset,
            entropy_coef=config['entropy_coef'],
            epochs = config['epochs']
    )
            
    agent.update(transition_list,agent_index)

    _return = 0
    for t in reversed(range(0, int(model.num_future_steps/offset))):
        _return = (config['gamma'] * _return) + float(transition_list[agent_index]['rewards'][t])
    cumulative_reward[agent_index]['return'].append(_return)

torch.save(agent.state_dict(), f'~/Multi-agent-competitive-environment/checkpoints/ppo_episodes={episodes}.ckpt')
vis_reward(new_data,cumulative_reward,agent_index,episodes)