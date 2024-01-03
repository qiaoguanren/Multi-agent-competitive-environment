import pytorch_lightning as pl
import cv2
import matplotlib.pyplot as plt
import os
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
from visualization.vis import vis_reward, generate_video, vis_adcr
from utils.geometry import wrap_angle
from utils.utils import get_transform_mat, get_auto_pred, add_new_agent, reward_function
from torch_geometric.data import Batch
from tqdm import tqdm

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
parser.add_argument("--ppo_config", default="PPO_epoch100.yaml", type=str)
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

with open("configs/"+args.ppo_config, "r") as file:
        config = yaml.safe_load(file)
file.close()

max_epoch = 0
base_path = 'figures/'
if os.path.exists(base_path):
    for folder in os.listdir(base_path):
        if folder.startswith("version_"):
            max_epoch+=1

next_version_folder = f"version_{max_epoch + 1}/"
next_version_path = os.path.join(base_path, next_version_folder)
os.makedirs(next_version_path, exist_ok=True)

for episode in tqdm(range(config['episodes'])):
    offset=5

    transition_list = [{'datas': [],
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []} for _ in range(config['buffer_batchsize'])]
        
    agent_index = torch.nonzero(data['agent']['category']==3,as_tuple=False).item()

    agent = PPO(
            state_dim=model.num_historical_steps*config['hidden_dim'],
            encoder = model.encoder,
            decoder = model.decoder,
            agent_index = agent_index,
            hidden_dim = config['hidden_dim'],
            action_dim = model.output_dim+1,
            batchsize = config['sample_batchsize'],
            actor_learning_rate = config['actor_learning_rate'],
            critic_learning_rate = config['critic_learning_rate'],
            lamda = config['lamda'],
            eps = config['eps'],
            gamma = config['gamma'],
            device = model.device,
            agent_num = new_input_data['agent']['num_nodes'],
            offset = offset,
            entropy_coef=config['entropy_coef'],
            epochs = config['epochs']
    )

    cumulative_reward = [{'return': []} for _ in range(new_input_data['agent']['num_nodes'])]

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

        state = model.encoder(new_data)

        for timestep in range(0,model.num_future_steps,offset):

            true_trans_position_refine=new_true_trans_position_refine
            reg_mask = new_data['agent']['predict_mask'][agent_index, model.num_historical_steps:]
            init_origin,init_theta,init_rot_mat=get_transform_mat(new_data,model)
            
            sample_action = agent.choose_action([state], model.decoder,[new_data],agent_index,offset)
            transition_list[batch]['datas'].append(new_data)

            best_mode = torch.randint(6,size=(data['agent']['num_nodes'],))
            best_mode = torch.cat([best_mode, torch.tensor([0])],dim=-1)
            l2_norm = (torch.norm(auto_pred['loc_refine_pos'][agent_index,:,:offset, :2] -
                                sample_action[:offset, :2].unsqueeze(0), p=2, dim=-1) * reg_mask[:offset].unsqueeze(0)).sum(dim=-1)
            action_suggest_index=l2_norm.argmin(dim=-1)
            best_mode[agent_index] = action_suggest_index

            new_data, auto_pred, _, _, (new_true_trans_position_propose, new_true_trans_position_refine),(traj_propose, traj_refine) = get_auto_pred(
                new_data, model, auto_pred["loc_refine_pos"][torch.arange(traj_propose.size(0)),best_mode], auto_pred["loc_refine_head"][torch.arange(traj_propose.size(0)),best_mode,:,0],offset,anchor=(init_origin,init_theta,init_rot_mat)
            )

            next_state = model.encoder(new_data)
            transition_list[batch]['states'].append(state)
            transition_list[batch]['actions'].append(sample_action[:offset,:])
            transition_list[batch]['next_states'].append(next_state)
            if timestep == model.num_future_steps - offset:
                transition_list[batch]['dones'].append(torch.tensor(1).cuda())
            else:
                transition_list[batch]['dones'].append(torch.tensor(0).cuda())
            reward = reward_function(new_input_data.clone(), new_data.clone(), model, agent_index, timestep)
            transition_list[batch]['rewards'].append(torch.tensor(reward).cuda())
            state = next_state
            
    agent.update(transition_list, config['buffer_batchsize'])

    discounted_return = 0
    undiscounted_return = 0
    adcr = []
    for t in reversed(range(0, int(model.num_future_steps/offset))):
        discounted_return = (config['gamma'] * discounted_return) + float(transition_list[-1]['rewards'][t])
        undiscounted_return += float(transition_list[-1]['rewards'][t])
        if episode % 100 == 0:
             adcr.append(discounted_return)
    cumulative_reward[0]['return'].append(discounted_return)
    cumulative_reward[1]['return'].append(undiscounted_return)

    if episode % 100 == 0:
        adcr.reverse()
        vis_adcr(adcr,next_version_path,offset)

vis_reward(new_data,cumulative_reward,0,config['episodes'],next_version_path)
vis_reward(new_data,cumulative_reward,1,config['episodes'],next_version_path)

pi_state_dict = agent.pi.state_dict()
old_pi_state_dict = agent.old_pi.state_dict()
value_state_dict = agent.value.state_dict()
model_state_dict = {
    'pi': pi_state_dict,
    'old_pi': old_pi_state_dict,
    'value': value_state_dict
}

base_path = 'checkpoints/'
next_version_path = os.path.join(base_path, next_version_folder)
os.makedirs(next_version_path, exist_ok=True)
torch.save(model_state_dict, next_version_path+'PPO_episodes='+str(config['episodes'])+'_epochs='+str(config['epochs'])+'.ckpt')