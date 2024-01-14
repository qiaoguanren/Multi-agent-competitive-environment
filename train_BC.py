import pytorch_lightning as pl
import cv2
import matplotlib.pyplot as plt
import os
import torch, math
import yaml
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser
from datasets import ArgoverseV2Dataset
from predictors import QCNet
from predictors.autoval import AntoQCNet
from predictors.environment import WorldModel
from transforms import TargetBuilder
from pathlib import Path
from PPO.actor_BC import BC
from utils.utils import get_transform_mat, get_auto_pred, add_new_agent,sample_from_pdf
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
parser.add_argument("--ppo_config", default="BC_epoch500.yaml", type=str)
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

new_input_data=add_new_agent(data)

with open("configs/"+args.ppo_config, "r") as file:
        config = yaml.safe_load(file)
file.close()

for episode in tqdm(range(config['episodes'])):
    offset=config['offset']

    transition_list = [{
                'states': [],
                'actions': []
                } for _ in range(config['buffer_batchsize'])]
        
    agent_index = torch.nonzero(data['agent']['category']==3,as_tuple=False).item()

    agent = BC(
            state_dim=model.num_modes*config['hidden_dim'],
            action_dim = model.output_dim*offset,
            hidden_dim = config['hidden_dim'],
            lr = config['actor_learning_rate'],
            epochs = config['epochs'],
            batchsize = config['sample_batchsize'],
            offset = offset
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
        pi = pred['pi']
        eval_mask = new_data['agent']['category'] == 3
        pi_eval = F.softmax(pi[eval_mask], dim=-1)

        state = environment.decoder(new_data, environment.encoder(new_data))[agent_index]

        for timestep in range(0,model.num_future_steps,offset):

            true_trans_position_refine=new_true_trans_position_refine
            reg_mask = new_data['agent']['predict_mask'][agent_index, model.num_historical_steps:]

            # max_value = -1
            # max_index = -1
            # for k in range(6):
            #     x = traj_propose[new_data["agent"]["category"] == 3, k, offset, 0].cpu()
            #     y = traj_propose[new_data["agent"]["category"] == 3, k, offset, 1].cpu()
            #     value = np.sqrt(x**2 + y**2)
            #     if value > max_value:
            #         max_value = value
            #         max_index = k\
            best_mode = [sample_from_pdf(pi_eval) for _ in range(new_input_data['agent']['num_nodes'])]
            best_mode = torch.tensor(np.array(best_mode))

            # best_mode = torch.randint(6,size=(new_input_data['agent']['num_nodes'],))
            action = auto_pred["loc_refine_pos"][agent_index,best_mode[agent_index],:offset,:2]
            action = action.flatten(start_dim = 1)

            new_data, auto_pred, _, _, (new_true_trans_position_propose, new_true_trans_position_refine),(traj_propose, traj_refine) = get_auto_pred(
                new_data, model, auto_pred["loc_refine_pos"][torch.arange(traj_propose.size(0)),best_mode], auto_pred["loc_refine_head"][torch.arange(traj_propose.size(0)),best_mode,:,0],offset,anchor=(init_origin,init_theta,init_rot_mat)
            )

            next_state = environment.decoder(new_data, environment.encoder(new_data))[agent_index].detach()
            transition_list[batch]['states'].append(state)
            transition_list[batch]['actions'].append(action)
            state = next_state
            pi_eval = F.softmax(auto_pred['pi'][eval_mask], dim=-1)
            
    agent.train(transition_list, config['buffer_batchsize'])

pi_state_dict = agent.actor.state_dict()
model_state_dict = {
    'actor': pi_state_dict,
}

base_path = 'checkpoints/'
torch.save(model_state_dict, base_path+'BC_episodes='+str(config['episodes'])+'_epochs='+str(config['epochs'])+'.ckpt')

