import torch, copy, math, os, random
import numpy as np
import csv
from utils.geometry import wrap_angle
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import torch.nn.functional as F

def get_transform_mat(input_data,model):
    origin = input_data["agent"]["position"][:, model.num_historical_steps - 1]
    theta = input_data["agent"]["heading"][:, model.num_historical_steps - 1]
    cos, sin = theta.cos(), theta.sin()
    rot_mat = theta.new_zeros(input_data["agent"]["num_nodes"], 2, 2)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = -sin
    rot_mat[:, 1, 0] = sin
    rot_mat[:, 1, 1] = cos
    return origin,theta,rot_mat

def get_auto_pred(input_data, model, loc_refine_pos, loc_refine_head, offset, anchor=None):
    old_anchor=origin,theta,rot_mat=get_transform_mat(input_data,model)
    # auto_index = data['agent']['valid_mask'][:,model.num_historical_steps]
    input_data["agent"]["valid_mask"] = (
        torch.cat(
            (
                input_data["agent"]["valid_mask"][..., offset:],
                torch.zeros(input_data["agent"]["valid_mask"].shape[:-1] + (5,)).cuda(),
            ),
            dim=-1,
        )
    ).bool()
    input_data["agent"]["valid_mask"][:, 0] = False
    new_position = torch.matmul(
        loc_refine_pos[..., :2], rot_mat.swapaxes(-1, -2)
    ) + origin[:, :2].unsqueeze(1)
    input_position = torch.zeros_like(input_data["agent"]["position"])
    input_position[:, :-offset] = input_data["agent"]["position"][:, offset:]
    input_position[
        :, model.num_historical_steps - offset : model.num_historical_steps, :2
    ] = new_position[:, :offset]

    input_v = torch.zeros_like(input_data["agent"]["velocity"])
    input_v[:, :-offset] = input_data["agent"]["velocity"][:, offset:]
    input_v[:, model.num_historical_steps - offset : model.num_historical_steps, :2] = (
        new_position[:, 1:] - new_position[:, :-1]
    )[:, :offset] / 0.1
    
    

    input_heading = torch.zeros_like(input_data["agent"]["heading"])
    input_heading[:, :-offset] = input_data["agent"]["heading"][:, offset:]
    input_heading[
        :, model.num_historical_steps - offset : model.num_historical_steps
    ] = wrap_angle(loc_refine_head+theta.unsqueeze(-1))[:,:offset]
    input_data["agent"]["position"] = input_position
    input_data["agent"]["heading"] = input_heading
    input_data["agent"]["velocity"] = input_v

    auto_pred = model(input_data)
    new_anchor=get_transform_mat(input_data,model)
    def get_transform_res(old_anchor,new_anchor,auto_pred):
        old_origin,old_theta,old_rot_mat=old_anchor
        new_origin,new_theta,new_rot_mat=new_anchor
        new_trans_position_propose = torch.einsum(
            "bijk,bkn->bijn",
            auto_pred["loc_propose_pos"][..., : model.output_dim],
            new_rot_mat.swapaxes(-1, -2),
        ) + new_origin[:, :2].unsqueeze(1).unsqueeze(1)
        new_pred=copy.deepcopy(auto_pred)
        new_pred["loc_propose_pos"][..., : model.output_dim] = torch.einsum(
            "bijk,bkn->bijn",
            new_trans_position_propose.cuda() - old_origin[:, :2].unsqueeze(1).unsqueeze(1).cuda(),
            old_rot_mat.cuda(),
        )
        new_pred["scale_propose_pos"][..., model.output_dim - 1] = wrap_angle(
            auto_pred["scale_propose_pos"][..., model.output_dim - 1].cuda()
            + new_theta.unsqueeze(-1).unsqueeze(-1).cuda()
            - old_theta.unsqueeze(-1).unsqueeze(-1).cuda()
        )

        new_trans_position_refine = torch.einsum(
            "bijk,bkn->bijn",
            auto_pred["loc_refine_pos"][..., : model.output_dim].cuda(),
            new_rot_mat.swapaxes(-1, -2).cuda(),
        ) + new_origin[:, :2].unsqueeze(1).unsqueeze(1).cuda()
        new_pred["loc_refine_pos"][..., : model.output_dim] = torch.einsum(
            "bijk,bkn->bijn",
            new_trans_position_refine.cuda() - old_origin[:, :2].unsqueeze(1).unsqueeze(1).cuda(),
            old_rot_mat.cuda(),
        )
        new_pred["scale_refine_pos"][..., model.output_dim - 1] = wrap_angle(
            auto_pred["scale_refine_pos"][..., model.output_dim - 1].cuda()
            + new_theta.unsqueeze(-1).unsqueeze(-1).cuda()
            - old_theta.unsqueeze(-1).unsqueeze(-1).cuda()
        )
        return new_pred,(new_trans_position_propose, new_trans_position_refine),

    _,(new_trans_position_propose, new_trans_position_refine)=get_transform_res(old_anchor,new_anchor,auto_pred)
    if model.output_head:
        auto_traj_propose = torch.cat(
            [
                auto_pred["loc_propose_pos"][..., : model.output_dim],
                auto_pred["loc_propose_head"],
                auto_pred["scale_propose_pos"][..., : model.output_dim],
                auto_pred["conc_propose_head"],
            ],
            dim=-1,
        )
        auto_traj_refine = torch.cat(
            [
                auto_pred["loc_refine_pos"][..., : model.output_dim],
                auto_pred["loc_refine_head"],
                auto_pred["scale_refine_pos"][..., : model.output_dim],
                auto_pred["conc_refine_head"],
            ],
            dim=-1,
        )
    else:
        auto_traj_propose = torch.cat([auto_pred['loc_propose_pos'][..., :model.output_dim],
                                auto_pred['scale_propose_pos'][..., :model.output_dim]], dim=-1)
        auto_traj_refine = torch.cat([auto_pred['loc_refine_pos'][..., :model.output_dim],
                                auto_pred['scale_refine_pos'][..., :model.output_dim]], dim=-1)
    if anchor is not None:
        anchor_auto_pred,_=get_transform_res(anchor,new_anchor,auto_pred)
        if model.output_head:
            anchor_auto_traj_propose = torch.cat([anchor_auto_pred['loc_propose_pos'][..., :model.output_dim],
                                                  anchor_auto_pred["loc_propose_head"],
                                    anchor_auto_pred['scale_propose_pos'][..., :model.output_dim],
                                    anchor_auto_pred["conc_propose_head"],], dim=-1)
            anchor_auto_traj_refine = torch.cat([anchor_auto_pred['loc_refine_pos'][..., :model.output_dim],
                                                 anchor_auto_pred["loc_refine_head"],
                                    anchor_auto_pred['scale_refine_pos'][..., :model.output_dim],
                                    anchor_auto_pred["conc_refine_head"],], dim=-1)
        else:
            anchor_auto_traj_propose = torch.cat([anchor_auto_pred['loc_propose_pos'][..., :model.output_dim],
                                    anchor_auto_pred['scale_propose_pos'][..., :model.output_dim]], dim=-1)
            anchor_auto_traj_refine = torch.cat([anchor_auto_pred['loc_refine_pos'][..., :model.output_dim],
                                    anchor_auto_pred['scale_refine_pos'][..., :model.output_dim]], dim=-1)

    return (
        input_data,
        auto_pred,
        auto_traj_refine,
        auto_traj_propose,
        (new_trans_position_propose, new_trans_position_refine),
        None if anchor is None else (anchor_auto_traj_propose, anchor_auto_traj_refine)
    )

def add_new_agent(data, a, v0_x, v0_y, heading, x0, y0):
    acceleration = a
    arr_s_x = np.array([])
    arr_s_y = np.array([])
    arr_v_x = np.array([])
    arr_v_y = np.array([])
    # v0_x = 1*math.cos(1.23)
    t = 0.1
    x = x0
    y = y0
    # x0 = x = 5259.7
    # y0 = y = 318
    # x0 = x = 2665
    # y0 = y = -2410
    v_x = 0
    v_y = 0
    new_heading=torch.empty_like(data['agent']['heading'][0])
    # new_heading[:]=1.9338
    # new_heading[:]=0.3898
    new_heading[:]=heading


    for i in range(110):

        a_x = acceleration*math.cos(new_heading[i])
        x = x + v0_x*t + 0.5*acceleration*(t**2)
        v0_x = v0_x + a_x*t
        v_x = v0_x
        arr_s_x = np.append(arr_s_x,x)
        arr_v_x = np.append(arr_v_x,v_x)

        if v0_y < 0:
            a_y = -math.sqrt(acceleration**2-a_x**2)
        else:
            a_y = math.sqrt(acceleration**2-a_x**2)
        y = y + v0_y*t + 0.5*acceleration*(t**2)
        v0_y = v0_y + a_y*t
        v_y = v0_y
        arr_s_y = np.append(arr_s_y,y)
        arr_v_y = np.append(arr_v_y,v_y)

    new_position=torch.empty_like(data['agent']['position'][0])
    new_position[:,0]=torch.tensor(np.concatenate([arr_s_x]))
    new_position[:,1]=torch.tensor(np.concatenate([arr_s_y]))

    new_velocity=torch.empty_like(data['agent']['velocity'][0])
    new_velocity[:,0]=torch.tensor(np.concatenate([arr_v_x]))
    new_velocity[:,1]=torch.tensor(np.concatenate([arr_v_y]))

    data=data.clone()
    data['agent']['num_nodes']+=1   #num_nodes
    #av_index
    data['agent']['valid_mask']=torch.cat([data['agent']['valid_mask'],torch.ones_like(data['agent']['valid_mask'][[0]])]) #valid_mask
    data['agent']['predict_mask']=torch.cat([data['agent']['predict_mask'],torch.ones_like(data['agent']['predict_mask'][[0]])]) #predict_mask
    data['agent']['id'][0].append(str(max(map(int,filter(str.isdigit,data['agent']['id'][0])))+1)) #id
    data['agent']['type']=torch.cat([data['agent']['type'],torch.tensor([0])]) #type
    data['agent']['category']=torch.cat([data['agent']['category'],torch.tensor([2])]) #category
    data['agent']['position']=torch.cat([data['agent']['position'],new_position[None,:]]) #position
    data['agent']['heading']=torch.cat([data['agent']['heading'],new_heading[None,:]]) #heading
    data['agent']['velocity']=torch.cat([data['agent']['velocity'],new_velocity[None,:]]) #velocity
    #target'
    data['agent']['batch']=torch.cat([data['agent']['batch'],torch.tensor([0])]) #batch
    data['agent']['ptr'][1]+=1    #ptr
    return data

def reward_function(data,new_data,model,agent_index):
                
        reward = 0

        gt = data['agent']['position'][agent_index, model.num_historical_steps+model.num_future_steps-1:model.num_historical_steps+model.num_future_steps, :model.output_dim]
        start_point = data['agent']['position'][agent_index, model.num_historical_steps-1:model.num_historical_steps, :model.output_dim]
        # gt = torch.zeros(1,2).cuda()
        # gt[0,0] = 5283
        # gt[0,1] = 306
        current_position = new_data['agent']['position'][agent_index, model.num_historical_steps-1:model.num_historical_steps, :model.output_dim]
        # pre_position = new_data['agent']['position'][agent_index, model.num_historical_steps-2:model.num_historical_steps-1, :model.output_dim]

        # delta_distance = gt-current_position
        l1_norm_current_distance = torch.norm(current_position - gt, p=1, dim=-1)
        # l2_norm_pre_distance = torch.norm(gt - pre_position, p=2, dim=-1)
        total_distance = torch.norm(gt - start_point, p=1, dim=-1)
        travel_distance = torch.norm(current_position - start_point, p=1, dim=-1)
        if travel_distance<total_distance:
            reward = math.log(travel_distance/total_distance)
        else:
            reward = l1_norm_current_distance
        
        return reward

def cost_function(new_data,model,agent_index):
    cost_limit = 15
    max_distance = 0
    for i in range(new_data['agent']['num_nodes']):
        if i==agent_index:
            continue
        distance = torch.norm(new_data['agent']['position'][agent_index, model.num_historical_steps-1:model.num_historical_steps, :model.output_dim]-new_data['agent']['position'][i, model.num_historical_steps-1:model.num_historical_steps, :model.output_dim],p=2, dim=-1)
        if distance > max_distance:
            max_distance = distance
    if max_distance >= cost_limit:
        return 0
    else:
        return max_distance - cost_limit

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dir(base_path):
    max_epoch = 0
    if os.path.exists(base_path):
        for folder in os.listdir(base_path):
            if folder.startswith("version_"):
                max_epoch+=1

    next_version_folder = f"version_{max_epoch + 1}/"
    next_version_path = os.path.join(base_path, next_version_folder)
    os.makedirs(next_version_path, exist_ok=True)
    return next_version_path

def save_reward(config_name, figure_folder, data_list, agent_number):
    
    file_path = '/home/guanren/Multi-agent-competitive-environment/'+figure_folder+config_name+'_agent-number'+str(agent_number)+'.csv'

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        if agent_number > 1:

            for row in data_list:
                writer.writerow(row)

        else:
            for data in data_list:
                writer.writerow([data])

def process_batch(batch, config, new_input_data, model, agents, choose_agent, noise, offset, transition_list):
    
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
        global_state = pred['first_m']
        for i in range(config['agent_number']):
            state_temp_list.append(global_state[choose_agent[i]])

        for timestep in range(0,model.num_future_steps,offset):

            # true_trans_position_refine=new_true_trans_position_refine
            # reg_mask_list = []
            # for i in range(config['agent_number']):
            #     reg_mask = new_data['agent']['predict_mask'][choose_agent[i], model.num_historical_steps:]
            #     reg_mask_list.append(reg_mask)
            
            best_mode = pi_eval.argmax(dim=-1)

            magnet_list = []
            action_list = []

            for i in range(config['agent_number']):
                scale,sample_action, log_prob = agents[i].choose_action(state_temp_list[i], noise)
                sample_action = sample_action.squeeze(0).reshape(-1,model.output_dim+1)

                auto_pred['loc_refine_pos'][choose_agent[i],best_mode[choose_agent[i]], :offset, :model.output_dim] = sample_action[..., :model.output_dim]
                auto_pred['loc_refine_head'][choose_agent[i],best_mode[choose_agent[i]], :offset] = sample_action[..., model.output_dim].unsqueeze(-1)
                auto_pred['scale_refine_pos'][choose_agent[i],best_mode[choose_agent[i]], :offset, :model.output_dim] = scale[..., :model.output_dim]
                auto_pred['conc_refine_head'][choose_agent[i],best_mode[choose_agent[i]], :offset] = scale[..., model.output_dim].unsqueeze(-1)
                
                action_list.append(sample_action)
                magnet_list.append(log_prob)

            # action_suggest_index_list = []
            # for i in range(config['agent_number']):
            #     l2_norm = (torch.norm(auto_pred['loc_refine_pos'][choose_agent[i],:,:offset, :2] -
            #                         sample_action_list[i][:offset, :2].unsqueeze(0), p=2, dim=-1) * reg_mask_list[i][timestep:timestep+offset].unsqueeze(0)).sum(dim=-1)
            #     action_suggest_index=l2_norm.argmin(dim=-1)
            #     best_mode[choose_agent[i]] = action_suggest_index
            #     action_suggest_index_list.append(action_suggest_index)

            new_data, auto_pred, _, _, (new_true_trans_position_propose, new_true_trans_position_refine),(traj_propose, traj_refine) = get_auto_pred(
                new_data, model, auto_pred["loc_refine_pos"][torch.arange(traj_propose.size(0)),best_mode], auto_pred["loc_refine_head"][torch.arange(traj_propose.size(0)),best_mode,:,0],offset,anchor=(init_origin,init_theta,init_rot_mat)
            )

            next_state_temp_list = []
            global_next_state = auto_pred['first_m']
            for i in range(config['agent_number']):
                next_state_temp_list.append(global_next_state[choose_agent[i]])
            for i in range(config['agent_number']):
                transition_list[batch]['states'][i].append(state_temp_list[i])
                transition_list[batch]['actions'][i].append(action_list[i])
                transition_list[batch]['next_states'][i].append(next_state_temp_list[i])
                transition_list[batch]['magnet'][i].append(magnet_list[i])
            if timestep == model.num_future_steps - offset:
                transition_list[batch]['dones'].append(torch.tensor(1).cuda())
            else:
                transition_list[batch]['dones'].append(torch.tensor(0).cuda())
            for i in range(config['agent_number']):
                reward = reward_function(new_input_data.clone(),new_data.clone(),model,choose_agent[i])
                transition_list[batch]['rewards'][i].append(torch.tensor([reward]).cuda())
                cost = cost_function(new_data.clone(),model,choose_agent[i])
                transition_list[batch]['costs'][i].append(torch.tensor([cost]).cuda())
                     
            state_temp_list = next_state_temp_list
            pi_eval = F.softmax(auto_pred['pi'], dim=-1)