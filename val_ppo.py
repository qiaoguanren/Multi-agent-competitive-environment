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
from predictors.environment import WorldModel
from transforms import TargetBuilder
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
    TrackCategory,
)
from av2.map.map_api import ArgoverseStaticMap
from pathlib import Path
from visualization.vis import vis_reward, generate_video, plot_traj_with_data
from utils.geometry import wrap_angle
from utils.utils import get_transform_mat, get_auto_pred, add_new_agent, reward_function
from torch_geometric.data import Batch
from PIL import Image as img
from tqdm import tqdm
from datetime import datetime

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

if isinstance(data, Batch):
    data['agent']['av_index'] += data['agent']['ptr'][:-1]

scenario_id = data["scenario_id"][0]
argoverse_scenario_dir = Path("/home/guanren/Multi-agent-competitive-environment/datasets/val/raw")

all_scenario_files = sorted(argoverse_scenario_dir.rglob(f"*_{scenario_id}.parquet"))
scenario_file_list = list(all_scenario_files)
scenario_path = scenario_file_list[0]
static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
scenario_static_map = ArgoverseStaticMap.from_json(static_map_path)

new_input_data=add_new_agent(data)

current_time = datetime.now()
timestamp = current_time.strftime("%Y%m%d_%H%M%S")
vid_path = f'videos/test_ppo_{timestamp}.webm'
with open("configs/PPO_epoch100.yaml", "r") as file:
        config = yaml.safe_load(file)
file.close()

model_state_dict = torch.load('checkpoints/version_4/PPO_episodes=500_epochs=100.ckpt')

episodes = 10

with torch.no_grad():
    for episode in tqdm(range(episodes)):
        offset=config['offset']

        transition_list = [{
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []} for _ in range(config['buffer_batchsize'])]

        agent_index = torch.nonzero(new_input_data['agent']['category']==3,as_tuple=False).item()
                
        agent = PPO(
            state_dim=model.num_modes*config['hidden_dim'],
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
        agent.pi.load_state_dict(model_state_dict['pi'])
        agent.old_pi.load_state_dict(model_state_dict['old_pi'])
        agent.value.load_state_dict(model_state_dict['value'])
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
            frames = []
            state = environment.decoder(new_data, environment.encoder(new_data))[agent_index]
            for i in range(40,110):
                  if i<50:
                      if episode == episodes - 1 and batch == config['buffer_batchsize']-1:
                          plot_traj_with_data(new_data,scenario_static_map,bounds=50,t=i)
                  else:
                      if i%offset==0:
                          true_trans_position_refine=new_true_trans_position_refine
                          reg_mask = new_data['agent']['predict_mask'][agent_index, model.num_historical_steps:]
                          
                          sample_action = agent.choose_action(state)

                          best_mode = torch.randint(6,size=(new_input_data['agent']['num_nodes'],))
                          l2_norm = (torch.norm(auto_pred['loc_refine_pos'][agent_index,:,:offset, :2] -
                                              sample_action[:offset, :2].unsqueeze(0), p=2, dim=-1) * reg_mask[:offset].unsqueeze(0)).sum(dim=-1)
                          action_suggest_index=l2_norm.argmin(dim=-1)
                          print(action_suggest_index)
                          best_mode[agent_index] = action_suggest_index

                          new_data, auto_pred, _, _, (new_true_trans_position_propose, new_true_trans_position_refine),(traj_propose, traj_refine) = get_auto_pred(
                              new_data, model, auto_pred["loc_refine_pos"][torch.arange(traj_propose.size(0)),best_mode], auto_pred["loc_refine_head"][torch.arange(traj_propose.size(0)),best_mode,:,0],offset,anchor=(init_origin,init_theta,init_rot_mat)
                          )

                          next_state = environment.decoder(new_data, environment.encoder(new_data))[agent_index].detach()
                          transition_list[batch]['states'].append(state)
                          transition_list[batch]['actions'].append(sample_action)
                          transition_list[batch]['next_states'].append(next_state)
                          if i == model.num_future_steps + model.num_historical_steps - offset:
                              transition_list[batch]['dones'].append(torch.tensor(1).cuda())
                          else:
                              transition_list[batch]['dones'].append(torch.tensor(0).cuda())
                          reward = reward_function(new_input_data.clone(), new_data.clone(), model, agent_index)
                          transition_list[batch]['rewards'].append(reward.clone())
                          state = next_state
                          if episode == episodes - 1 and batch == config['buffer_batchsize']-1:
                              plot_traj_with_data(new_data,scenario_static_map,bounds=50,t=50-offset)
                              for j in range(6):
                                xy = true_trans_position_refine[new_data["agent"]["category"] == 3][0].cpu()
                                plt.plot(xy[j, ..., 0], xy[j, ..., 1])
                      else:
                          if episode == episodes - 1 and batch == config['buffer_batchsize']-1:
                              plot_traj_with_data(new_data,scenario_static_map,bounds=50,t=50-offset+i%offset)
                              for j in range(6):
                                xy = true_trans_position_refine[new_data["agent"]["category"] == 3][0].cpu()
                                plt.plot(xy[j, i%offset:, 0], xy[j, i%offset:, 1])

                  buf = io.BytesIO()
                  plt.savefig(buf, format="png")
                  plt.close()
                  buf.seek(0)
                  frame = img.open(buf)
                  frames.append(frame)
        agent.update(transition_list, agent_index)
    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    video = cv2.VideoWriter(vid_path, fourcc, fps=10, frameSize=frames[0].size)
    for i in range(len(frames)):
        frame_temp = frames[i].copy()
        video.write(cv2.cvtColor(np.array(frame_temp), cv2.COLOR_RGB2BGR))
    video.release()

# vis_reward(new_data,cumulative_reward,agent_index,episodes)