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
from predictors.environment import WorldModel
from transforms import TargetBuilder
from pathlib import Path
from visualization.vis import vis_reward
from utils.geometry import wrap_angle
from utils.utils import get_transform_mat, get_auto_pred, add_new_agent, reward_function
from torch_geometric.data import Batch
from tqdm import tqdm

pl.seed_everything(2023, workers=True)

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="QCNet")
parser.add_argument("--root", type=str, default="/data2/guanren/qcnet_datasets")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--pin_memory", type=bool, default=True)
parser.add_argument("--persistent_workers", type=bool, default=True)
parser.add_argument("--accelerator", type=str, default="auto")
parser.add_argument("--devices", type=int, default=2)
parser.add_argument("--ckpt_path", default="checkpoints/epoch=10-step=274879.ckpt", type=str)
parser.add_argument("--ppo_config", default="PPO_epoch100.yaml", type=str)
args = parser.parse_args("")

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
for param in environment.encoder.parameters():
        param.requires_grad = False
for param in environment.decoder.parameters():
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

cumulative_reward = [[] for _ in range(new_input_data['agent']['num_nodes'])]

for episode in tqdm(range(config['episodes'])):
    offset=config['offset']

    transition_list = [{
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []} for _ in range(config['buffer_batchsize'])]
        
    agent_index = torch.nonzero(data['agent']['category']==3,as_tuple=False).item()

    agent = PPO(
            state_dim=model.num_modes*config['hidden_dim'],
            agent_index = agent_index,
            hidden_dim = config['hidden_dim'],
            action_dim = 3*offset,
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

        state = environment.decoder(new_data, environment.encoder(new_data))[agent_index]

        for timestep in range(0,model.num_future_steps,offset):

            true_trans_position_refine=new_true_trans_position_refine
            reg_mask = new_data['agent']['predict_mask'][agent_index, model.num_historical_steps:]
            
            sample_action = agent.choose_action(state)
            action = sample_action.flatten(start_dim = 1)
            min_val = action.min()
            max_val = action.max()
            normalized_tensor = (action - min_val) / (max_val - min_val)
            action = (2 * math.pi/2 * normalized_tensor - math.pi/2)

            best_mode = torch.randint(6,size=(data['agent']['num_nodes'],))
            best_mode = torch.cat([best_mode, torch.tensor([0])],dim=-1)
            l2_norm = (torch.norm(auto_pred['loc_refine_pos'][agent_index,:,:offset, :2] -
                                sample_action[:offset, :2].unsqueeze(0), p=2, dim=-1) * reg_mask[:offset].unsqueeze(0)).sum(dim=-1)
            action_suggest_index=l2_norm.argmin(dim=-1)
            best_mode[agent_index] = action_suggest_index

            new_data, auto_pred, _, _, (new_true_trans_position_propose, new_true_trans_position_refine),(traj_propose, traj_refine) = get_auto_pred(
                new_data, model, auto_pred["loc_refine_pos"][torch.arange(traj_propose.size(0)),best_mode], auto_pred["loc_refine_head"][torch.arange(traj_propose.size(0)),best_mode,:,0],offset,anchor=(init_origin,init_theta,init_rot_mat)
            )

            next_state = environment.decoder(new_data, environment.encoder(new_data))[agent_index].detach()
            transition_list[batch]['states'].append(state)
            transition_list[batch]['actions'].append(action)
            transition_list[batch]['next_states'].append(next_state)
            if timestep == model.num_future_steps - offset:
                transition_list[batch]['dones'].append(torch.tensor(1).cuda())
            else:
                transition_list[batch]['dones'].append(torch.tensor(0).cuda())
            reward = reward_function(new_input_data.clone(), new_data.clone(), model, agent_index)
            transition_list[batch]['rewards'].append(reward.clone())
            state = next_state
            
    agent.update(transition_list, config['buffer_batchsize'])

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

    cumulative_reward[agent_index].append(discounted_return)
    cumulative_reward[agent_index+1].append(undiscounted_return)

vis_reward(new_data,cumulative_reward,agent_index,config['episodes'],next_version_path)
# vis_reward(new_data,cumulative_reward,agent_index+1,config['episodes'],next_version_path)

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

