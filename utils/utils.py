import torch, copy, math
from utils.geometry import wrap_angle
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

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

def add_new_agent(data,new_position,new_heading,new_velocity):
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
    #target
    data['agent']['batch']=torch.cat([data['agent']['batch'],torch.tensor([0])]) #batch
    data['agent']['ptr'][1]+=1    #ptr
    return data

def reward_function(data,model,agent_index):
                
        reward1 = reward2 = reward3 = 0

        pre_v = data['agent']['velocity'][agent_index, model.num_historical_steps-2:model.num_historical_steps-1, :model.output_dim]
        next_v = data['agent']['velocity'][agent_index, model.num_historical_steps-1:model.num_historical_steps, :model.output_dim]

        pre_v = math.sqrt(pre_v[0,0]**2+pre_v[0,1]**2)
        next_v = math.sqrt(next_v[0,0]**2+next_v[0,1]**2)
        
        if next_v > pre_v:
             reward1 = 50
        else:
             reward1 = -100

        for i in range(data['agent']['num_nodes']):
            if i==agent_index:
                continue
            distance = torch.norm(data['agent']['position'][agent_index, model.num_historical_steps-1:model.num_historical_steps, :model.output_dim]-data['agent']['position'][i, model.num_historical_steps-1:model.num_historical_steps, :model.output_dim],dim=-1)
            if distance < 1e-1:
                 break
        if distance < 1e-1:
            reward2 = -100
        left_bound = []
        right_bound = []
        for i in range(len(data['map_point']['side'])):
            if data['map_point']['side'][i] == 0:
                left_bound.append(tuple(data['map_point']['position'][i,:2]))
            if data['map_point']['side'][i] == 1:
                right_bound.append(tuple(data['map_point']['position'][i,:2]))
        left_polygon = LineString(left_bound)
        right_polygon = LineString(right_bound)
        car_point = Point(tuple(data['agent']['position'][agent_index, model.num_historical_steps-1:model.num_historical_steps, :model.output_dim].flatten().cpu().numpy()))
        nearest_left = nearest_points(left_polygon, car_point)[0]
        nearest_right = nearest_points(right_polygon, car_point)[0]
        distance_to_nearest_left = car_point.distance(nearest_left)
        distance_to_nearest_right = car_point.distance(nearest_right)
        lane_width = nearest_left.distance(nearest_right)
        if distance_to_nearest_left + distance_to_nearest_right > lane_width:
            reward3 = -50

        # total_reward = np.array([reward1, reward2, reward3])
        # mean_reward = np.mean(total_reward)
        # std_reward = np.std(total_reward)    
        # normalized_reward = (total_reward - mean_reward) / std_reward
        # return float(np.sum(normalized_reward))
        return reward1+reward2+reward3