import pytorch_lightning as pl
import cv2
import matplotlib.pyplot as plt
import io
import torch, math
import random
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
from algorithm.masac import MASAC
from algorithm.mappo import MAPPO
from visualization.vis import vis_reward, generate_video, plot_traj_with_data
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from pathlib import Path
from utils.utils import get_transform_mat, get_auto_pred, add_new_agent, reward_function, sample_from_pdf
from torch_geometric.data import Batch
from PIL import Image as img
from tqdm import tqdm
from datetime import datetime

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="QCNet")
parser.add_argument("--root", type=str, default="/home/guanren/Multi-agent-competitive-environment/datasets")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--pin_memory", type=bool, default=True)
parser.add_argument("--persistent_workers", type=bool, default=True)
parser.add_argument("--accelerator", type=str, default="auto")
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--task", type=int, default=2)
parser.add_argument("--ckpt_path", default="checkpoints/epoch=10-step=274879.ckpt", type=str)
parser.add_argument("--RL_config", default="MASAC_eval", type=str)
args = parser.parse_args()

with open("configs/"+args.RL_config+'.yaml', "r") as file:
        config = yaml.safe_load(file)
file.close()
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

scenario_id = data["scenario_id"][0]
argoverse_scenario_dir = Path("/home/guanren/Multi-agent-competitive-environment/datasets/val/raw")

all_scenario_files = sorted(argoverse_scenario_dir.rglob(f"*_{scenario_id}.parquet"))
scenario_file_list = list(all_scenario_files)
scenario_path = scenario_file_list[0]

static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
scenario_static_map = ArgoverseStaticMap.from_json(static_map_path)

current_time = datetime.now()
timestamp = current_time.strftime("%Y%m%d_%H%M%S")
vid_path = f'videos/test_sac_{timestamp}.webm'

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
model_state_dict = torch.load('checkpoints/version_30/MASAC_episode500_epoch10_beta1e-1_seed1234_task2.ckpt')

offset=config['offset']
if config['algorithm'] != 'MASAC':
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

agent.actor.load_state_dict(model_state_dict['actor'])
agent.actor.eval()

with torch.no_grad():
    for episode in tqdm(range(config['episodes'])):
        scale = 0.0001
        offset=config['offset']

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
        pi = pred['pi']
        pi_eval = F.softmax(pi, dim=-1)
        state_temp_list = []
        global_state = environment.decoder(new_data, environment.encoder(new_data))
        for j in range(config['agent_number']):
            state_temp_list.append(global_state[choose_agent[j]])

        for i in range(40,110):
            if i<50:
                if episode == config['episodes'] - 1:
                    plot_traj_with_data(new_data,scenario_static_map,agent_number=config['agent_number'], bounds=80,t=i)
            else:
                if i%offset==0:
                    true_trans_position_refine=new_true_trans_position_refine
                    reg_mask_list = []
                    for j in range(config['agent_number']):
                        reg_mask = new_data['agent']['predict_mask'][choose_agent[j], model.num_historical_steps:]
                        reg_mask_list.append(reg_mask)
                    
                    sample_action_list = []
                    for j in range(config['agent_number']):
                        sample_action = agent.choose_action(state_temp_list[j], scale)
                        sample_action = sample_action.squeeze(0).reshape(-1,model.output_dim)
                        sample_action_list.append(sample_action)

                    best_mode = pi_eval.argmax(dim=-1)
                    for j in range(config['agent_number']):
                        l2_norm = (torch.norm(auto_pred['loc_refine_pos'][choose_agent[j],:,:offset, :2] -
                                            sample_action_list[j][:offset, :2].unsqueeze(0), p=2, dim=-1) * reg_mask_list[j][i-model.num_historical_steps:i-model.num_historical_steps+offset].unsqueeze(0)).sum(dim=-1)
                        action_suggest_index=l2_norm.argmin(dim=-1)
                        best_mode[choose_agent[j]] = action_suggest_index


                    new_data, auto_pred, _, _, (new_true_trans_position_propose, new_true_trans_position_refine),(traj_propose, traj_refine) = get_auto_pred(
                        new_data, model, auto_pred["loc_refine_pos"][torch.arange(traj_propose.size(0)),best_mode], auto_pred["loc_refine_head"][torch.arange(traj_propose.size(0)),best_mode,:,0],offset,anchor=(init_origin,init_theta,init_rot_mat)
                    )
                    next_state_temp_list = []
                    global_next_state = environment.decoder(new_data, environment.encoder(new_data))
                    for j in range(config['agent_number']):
                        next_state_temp_list.append(global_next_state[choose_agent[j]])

                    state_temp_list = next_state_temp_list
                      
                      # temp = new_data.clone()

                      # print(episode)

                      # for k in range(6):
                      #     print(math.sqrt(auto_pred['loc_refine_pos'][agent_index,k,offset-1, 0]**2+auto_pred['loc_refine_pos'][agent_index,k,offset-1, 1]**2))

                      #     best_mode[agent_index] = k
                      #     temp2, auto_pred, _, _, (new_true_trans_position_propose, new_true_trans_position_refine),(traj_propose, traj_refine) = get_auto_pred(
                      #         temp, model, auto_pred["loc_refine_pos"][torch.arange(traj_propose.size(0)),best_mode], auto_pred["loc_refine_head"][torch.arange(traj_propose.size(0)),best_mode,:,0],offset,anchor=(init_origin,init_theta,init_rot_mat)
                      #     )

                      #     if k == action_suggest_index:

                      #       new_data = temp2


                      # # new_data, auto_pred, _, _, (new_true_trans_position_propose, new_true_trans_position_refine),(traj_propose, traj_refine) = get_auto_pred(
                      # #     new_data, model, auto_pred["loc_refine_pos"][torch.arange(traj_propose.size(0)),best_mode], auto_pred["loc_refine_head"][torch.arange(traj_propose.size(0)),best_mode,:,0],offset,anchor=(init_origin,init_theta,init_rot_mat)
                      # # )

                      #       next_state = environment.decoder(temp2, environment.encoder(temp2))[agent_index]
                      #       state = next_state
                      #       print(agent.value(next_state.flatten(start_dim=0).unsqueeze(0)))
                      #     else:
                      #       next_state = environment.decoder(temp2, environment.encoder(temp2))[agent_index]
                      #       print(agent.value(next_state.flatten(start_dim=0).unsqueeze(0)))
                    pi_eval = F.softmax(auto_pred['pi'], dim=-1)
                    if episode == config['episodes'] - 1:
                        plot_traj_with_data(new_data,scenario_static_map,agent_number=config['agent_number'], bounds=80,t=50-offset)
                        for j in range(6):
                            xy = true_trans_position_refine[new_data["agent"]["category"] == 3][0].cpu()
                            plt.plot(xy[j, ..., 0], xy[j, ..., 1])
                else:
                    if episode == config['episodes'] - 1:
                        plot_traj_with_data(new_data,scenario_static_map,agent_number=config['agent_number'], bounds=80,t=50-offset+i%offset)
                        for j in range(6):
                            xy = true_trans_position_refine[new_data["agent"]["category"] == 3][0].cpu()
                            plt.plot(xy[j, i%offset:, 0], xy[j, i%offset:, 1])

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            frame = img.open(buf)
            frames.append(frame)

    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    video = cv2.VideoWriter(vid_path, fourcc, fps=10, frameSize=frames[0].size)
    for i in range(len(frames)):
        frame_temp = frames[i].copy()
        video.write(cv2.cvtColor(np.array(frame_temp), cv2.COLOR_RGB2BGR))
    video.release()

# vis_reward(new_data,cumulative_reward,agent_index,episodes)