import pytorch_lightning as pl
import cv2

import os, csv
import torch, math
import yaml
import numpy as np
import torch.nn.functional as F
from algorithm.mappo import MAPPO
from algorithm.masac import MASAC
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser
from datasets import ArgoverseV2Dataset
from predictors.autoval import AntoQCNet
from predictors.environment import WorldModel
from transforms import TargetBuilder
from utils.utils import add_new_agent, process_batch, save_reward, create_dir, save_gap
from torch_geometric.data import Batch
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="QCNet")
parser.add_argument("--root", type=str, default="/home/guanren/Multi-agent-competitive-environment/datasets")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--pin_memory", type=bool, default=True)
parser.add_argument("--persistent_workers", type=bool, default=True)
parser.add_argument("--accelerator", type=str, default="auto")
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--task", type=int, default=1)
parser.add_argument("--ckpt_path", default="checkpoints/epoch=10-step=274879.ckpt", type=str)
parser.add_argument("--RL_config", default="MASAC_episode500_epoch20_beta1e-1_seed1234", type=str)
args = parser.parse_args()

with open("configs/"+args.RL_config+'.yaml', "r") as file:
        config = yaml.safe_load(file)
file.close()
print(config['seed'])
pl.seed_everything(config['seed'], workers=True)
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

new_input_data=data
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
if 'MASAC' not in config['algorithm']:
    agent = MAPPO(
                state_dim=model.num_modes*config['hidden_dim'],
                action_dim = model.output_dim*offset*6,
                config = config,
                device = model.device,
                offset = offset
        )
else:
    agent = MASAC(
          state_dim=model.num_modes*config['hidden_dim'],
          action_dim = model.output_dim*offset*6,
          config = config,
          device = model.device,
          offset = offset
    )

choose_agent = []    
agent_index = torch.nonzero(data['agent']['category']==3,as_tuple=False).item()
choose_agent.append(agent_index)
for i in range(config['agent_number']-1):
    choose_agent.append(data['agent']['num_nodes']+i)

v_array = np.array([])

for episode in tqdm(range(config['episodes'])):
    
    scale=1/config['episodes']

    transition_list = [{
                'states': [[]for _  in range(config['agent_number'])],
                'actions': [[]for _  in range(config['agent_number'])],
                'next_states': [[]for _  in range(config['agent_number'])],
                'rewards': [[]for _  in range(config['agent_number'])],
                'dones': []} for _ in range(config['buffer_batchsize'])]
    
    for batch in range(config['buffer_batchsize']):
        process_batch(batch, config, new_input_data, model, environment, agent, choose_agent, scale, offset, transition_list)

    for i in range(config['agent_number']):
        v = agent.update(transition_list, config['buffer_batchsize'], scale, i)
        if i==0:
            print(v)
            v_array = np.append(v_array, v)

    discounted_return_list = []
    discounted_return = 0
    undiscounted_return = 0

    for i in range(config['agent_number']):
        for t in reversed(range(0, int(model.num_future_steps/offset))):
            _sum = 0
            for b in range(config['buffer_batchsize']):
                _sum+=float(transition_list[b]['rewards'][i][t])
            mean_reward = _sum/config['buffer_batchsize']
            discounted_return = (config['gamma'] * discounted_return) + mean_reward
            undiscounted_return += mean_reward
        discounted_return_list.append(discounted_return)
        print(discounted_return, undiscounted_return)
        discounted_return = 0
        undiscounted_return = 0

    cumulative_reward.append(discounted_return_list)

save_reward(args.RL_config+'_task'+str(args.task), next_version_path, cumulative_reward, config['agent_number'])
save_gap(args.RL_config+'_task'+str(args.task)+'_CCE-GAP', next_version_path, v_array.tolist())

if config['algorithm'] != 'MASAC':
    pi_state_dict = agent.pi.state_dict()
    old_pi_state_dict = agent.old_pi.state_dict()
    # old_value_state_dict = agent.old_value.state_dict()
    value_state_dict = agent.value.state_dict()
    model_state_dict = {
        'pi': pi_state_dict,
        # 'old_pi': old_pi_state_dict,
        # 'old_value': old_value_state_dict,
        'value': value_state_dict
    }
else:
    actor_state_dict = agent.actor.state_dict()
    critic_1_state_dict = agent.critic_1.state_dict()
    critic_2_state_dict = agent.critic_2.state_dict()
    model_state_dict = {
        'actor': actor_state_dict,
        'critic_1': critic_1_state_dict,
        'critic_2': critic_2_state_dict,
    }

next_version_path = create_dir(base_path = 'checkpoints/')
torch.save(model_state_dict, next_version_path+args.RL_config+'_task'+str(args.task)+'.ckpt')

