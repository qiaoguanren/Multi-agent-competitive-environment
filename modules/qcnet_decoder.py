# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse

from layers import AttentionLayer
from layers import FourierEmbedding
from layers import MLPLayer
from utils import angle_between_2d_vectors
from utils import bipartite_dense_to_sparse
from utils import weight_init
from utils import wrap_angle

class QCNetDecoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(QCNetDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        num_future_steps = 110
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_t2m_steps = num_t2m_steps if num_t2m_steps is not None else num_historical_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_r_t = 4
        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3

        self.mode_emb = nn.Embedding(num_modes, hidden_dim)
        self.r_t2m_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2m_emb = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2m_emb = FourierEmbedding(input_dim=input_dim_r_a2m, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.y_emb = FourierEmbedding(input_dim=output_dim + output_head, hidden_dim=hidden_dim,
                                      num_freq_bands=num_freq_bands)
        self.traj_emb = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                               batch_first=False, dropout=0.0, bidirectional=False)
        self.traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.t2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_propose_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                     dropout=dropout, bipartite=False, has_pos_emb=False)
        self.t2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_refine_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                    dropout=dropout, bipartite=False, has_pos_emb=False)
        self.to_loc_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=output_dim // num_recurrent_steps)
        self.to_scale_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                             output_dim=output_dim // num_recurrent_steps)
        self.to_loc_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                          output_dim=output_dim)
        self.to_scale_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            output_dim=output_dim)
        if output_head:
            self.to_loc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=1)
            self.to_conc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                 output_dim=1)
            self.to_loc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
            self.to_conc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=1)
        else:
            self.to_loc_propose_head = None
            self.to_conc_propose_head = None
            self.to_loc_refine_head = None
            self.to_conc_refine_head = None
        self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        self.apply(weight_init)

    def forward(self,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor],
                step = None) -> Dict[str, torch.Tensor]:

        # if step is not None:

        #     # num_total_steps = 110
        #     num_agents, num_total_steps = data['agent']['position'].shape[:2]
            
        #     pos_m = data['agent']['position'][:, step-1:step, :self.input_dim].reshape(-1, self.input_dim) # [N, 2] -> [N*step, 2]
        #     head_m = data['agent']['heading'][:, step-1:step].reshape(-1) # [N] -> [N*step]
        #     head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1) # [N, 2] -> [N, step, 2]
        #     total_num_nodes = pos_m.shape[0]

        #     # Time to m
        #     x_t = scene_enc['x_a'][:, :step].reshape(-1, self.hidden_dim) # -> [N*110, hidden]
        #     x_pl = scene_enc['x_pl'][:, :step].repeat(self.num_modes, 1, 1).reshape(-1, self.hidden_dim) # [N*num_modes, hiddem] -> [N*num_modes*110, hiddem]
        #     x_a = scene_enc['x_a'][:, :step].repeat(self.num_modes, 1, 1).reshape(-1, self.hidden_dim) # [N*num_modes, 128] (last position) -> [N*num_modes*110, 128]
        #     m = self.mode_emb.weight.repeat(scene_enc['x_a'].size(0)*1, 1) # [N*num_modes, 128] -> [N*num_modes*1, 128]

        #     mask_src = data['agent']['valid_mask'][:, :step].contiguous() # [N, 50] -> [N, 110] indicating avaliable positions for all agents
        #     # mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False # too old sources set to false
        #     mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True).repeat(1, self.num_modes) # [N, num_modes] -> [N, num_modes] indicating all agents
        #     # mask_dst = data['agent']['valid_mask'][:, step:step+1].contiguous() # all current avaliable nodes


        #     pos_t = data['agent']['position'][:, :step, :self.input_dim].reshape(-1, self.input_dim) # [N*50, 2] -> [N*step, 2]
        #     head_t = data['agent']['heading'][:, :step].reshape(-1) # [N*50] -> [N*110]

        #     # mask = data['agent']['valid_mask'][:, :step-1].contiguous()
        #     # mask_t = mask.unsqueeze(2) & mask.unsqueeze(1) # shape [N, 110, 110], (i, j) is True only when both i and j are true in mask (valid for both time step i and j)
        #     edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst[:, -1:].unsqueeze(1)) # [2, N1] all avaliable nodes pointing towards the current node
        #     # edge_index_t2m = dense_to_sparse(mask_t)[0] # sparse indexes of mask_t, of shape [2, N1], corresponds to the indexes of true values in mask_t
            
        #     # if the above are correct, the following line should make no difference
        #     # edge_index_t2m1 = edge_index_t2m[:, (edge_index_t2m % (step-1))[1] > (edge_index_t2m % (step-1))[0]] # only keep the time steps that j is larger than i (keeps half to avoid duplicated index)
            
        #     self.time_span = 10 
        #     edge_index_t2m = edge_index_t2m[:, edge_index_t2m[1] - edge_index_t2m[0] <= self.time_span] # ensures that edges that j - i < historical time steps
            
            
        #     # edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst[:, -1:].unsqueeze(1)) # [2, N1] all avaliable nodes pointing towards the current node -> all past nodes pointing towards current nodes
        #     rel_pos_t2m = pos_t[edge_index_t2m[0]] - pos_m[edge_index_t2m[1]] # [N1, 2] N1=number of avaliable edges from all avaliable nodes to current nodes

        #     rel_head_t2m = wrap_angle(head_t[edge_index_t2m[0]] - head_m[edge_index_t2m[1]]) # all avaliable headings to the current heading
        #     r_t2m = torch.stack(
        #         [torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1), 
        #         angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m[1]], nbr_vector=rel_pos_t2m[:, :2]),
        #         rel_head_t2m,
        #         (edge_index_t2m[0] % num_total_steps) - num_total_steps + 1], dim=-1)
        #     r_t2m = self.r_t2m_emb(continuous_inputs=r_t2m, categorical_embs=None)
        #     # edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1)) # [2, N1*num_modes]
        #     r_t2m = r_t2m.repeat_interleave(repeats=self.num_modes, dim=0) # [N1*num_modes, 128] repeated num_mode times


        #     # Polylines to m
        #     pos_pl = data['map_polygon']['position'][:, :self.input_dim] # [M1, 2]
        #     orient_pl = data['map_polygon']['orientation'] # [M1]

        #     # [2, N2] N2 = number of edges between polylines and current nodes
        #     # x: [N, 2]
        #     # y: [M1, 2]
        #     edge_index_pl2m = radius(
        #         x=pos_m[:, :2], 
        #         y=pos_pl[:, :2],
        #         r=self.pl2m_radius,
        #         batch_x=data['agent']['batch'].repeat_interleave(1) if isinstance(data, Batch) else None,
        #         batch_y=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
        #         max_num_neighbors=3000) # edge_index_pl2m[0] are indexes of y, edge_index_pl2m[0] are indexes of x
        #     # originally, there were pl pointing towards current node, now there are pl pointing towards all nodes
        #     # so the max neighbor limit is multiplied by 110
        #     edge_index_pl2m = edge_index_pl2m[:, mask_dst.reshape(-1)[edge_index_pl2m[1]]] # selecting only those whose current nodes are avaliable
        #     rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]] # [N2, 2] relative positions between polyline positions and current position
        #     rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]]) # [N2]
        #     r_pl2m = torch.stack( # [N2, 3]
        #         [torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
        #         angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]),
        #         rel_orient_pl2m], dim=-1)
        #     r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m, categorical_embs=None) # [N2, 128]
        #     edge_index_pl2m = torch.cat([edge_index_pl2m + i * edge_index_pl2m.new_tensor(
        #         [[data['map_polygon']['num_nodes']], [data['agent']['num_nodes']]]) for i in range(self.num_modes)], dim=1) # [2, N2*num_modes]
        #     r_pl2m = r_pl2m.repeat(self.num_modes, 1) # [N2*num_modes, 2]

        #     edge_index_a2m = radius_graph( # [2, N3] current nodes pointing towards each other
        #         x=pos_m[:, :2],
        #         r=self.a2m_radius,
        #         batch=data['agent']['batch'] if isinstance(data, Batch) else None,
        #         loop=False,
        #         max_num_neighbors=3000) # 110 times of original num of neighborhoods

        #     # This line should make no difference if the previous is correct
        #     edge_index_a2m = edge_index_a2m[:, edge_index_a2m[1] > edge_index_a2m[0]] # this is supposed to capture more information than before: before there is only edges in the current time span, now there are edges from history to current
            
        #     # create a time mask to select from the edge_index_a2m so that only current pointing to the current
        #     edge_index_a2m = edge_index_a2m[:, mask_src[:, -1:].reshape(-1)[edge_index_a2m[0]] & mask_dst.reshape(-1)[edge_index_a2m[1]]] # [2, N3'] select edges whose both side of nodes are valid
        #     rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]] # [N3', 2] relative positions of current nodes
        #     rel_head_a2m = wrap_angle(head_m[edge_index_a2m[0]] - head_m[edge_index_a2m[1]]) # [N3']
        #     r_a2m = torch.stack( # [N3, 3]
        #         [torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
        #         angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2m[1]], nbr_vector=rel_pos_a2m[:, :2]),
        #         rel_head_a2m], dim=-1)
        #     r_a2m = self.r_a2m_emb(continuous_inputs=r_a2m, categorical_embs=None)
        #     edge_index_a2m = torch.cat(
        #         [edge_index_a2m + i * edge_index_a2m.new_tensor([data['agent']['num_nodes']]) for i in
        #         range(self.num_modes)], dim=1)
        #     r_a2m = r_a2m.repeat(self.num_modes, 1)

        #     # edge_index_m2m = dense_to_sparse(mask_dst.reshape(-1, 1).unsqueeze(2) & mask_dst.reshape(-1, 1).unsqueeze(1))[0] # edges between current avaliable nodes
        #     # edge_index_m2m = dense_to_sparse(mask_dst.reshape(-1, 1).unsqueeze(2) & mask_dst.reshape(-1, 1).unsqueeze(1))[0] # edges between current avaliable nodes
        #     # edge_index_m2m = radius_graph( # [2, N3] current nodes pointing towards each other
        #     #     x=pos_m[:, :2],
        #     #     r=100000,
        #     #     batch=data['agent']['batch'].repeat_interleave(110) if isinstance(data, Batch) else None,
        #     #     loop=False,
        #     #     max_num_neighbors=110*110)

        #     edge_index_m2m = dense_to_sparse(mask_dst.unsqueeze(2) & mask_dst.unsqueeze(1))[0] # edges between current avaliable nodes
        #     # edge_index_m2m = dense_to_sparse(mask_dst.reshape(-1).unsqueeze(0).repeat(total_num_nodes, 1))[0] # edges between current avaliable nodes
        #     # edge_index_m2m = edge_index_m2m[:, edge_index_m2m[0]//num_total_steps == edge_index_m2m[1]//num_total_steps]

        #     locs_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        #     scales_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        #     locs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        #     concs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps

        #     for t in range(self.num_recurrent_steps):
        #         for i in range(self.num_layers):
        #             # import pdb; pdb.set_trace()
        #             m = m.reshape(-1, self.hidden_dim) # [N*num_modes, hidden_dim] -> [N*num_modes*110, hidden_dim]
        #             m = self.t2m_propose_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m) # integrate information of other agents and all time span to m
        #             m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        #             m = self.pl2m_propose_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m) # integrate the map info into m
        #             m = self.a2m_propose_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m) # integrate the info of other agents in the current time to m
        #             m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        #         m = self.m2m_propose_attn_layer(m, None, edge_index_m2m) # self attention
        #         m = m.reshape(-1, self.num_modes, self.hidden_dim) # [N, num_modes, 128] -> [N*num_steps, num_modes, 128]
        #         locs_propose_pos[t] = self.to_loc_propose_pos(m).reshape(num_agents, 1, 2) # [N*110, num_modes, 2]
        #         scales_propose_pos[t] = self.to_scale_propose_pos(m).reshape(num_agents, 1, 2)
        #         if self.output_head:
        #             locs_propose_head[t] = self.to_loc_propose_head(m).view(-1, self.num_modes, 1, 1)
        #             concs_propose_head[t] = self.to_conc_propose_head(m).view(-1, self.num_modes, 1, 1)
        #     loc_propose_pos = torch.cumsum(
        #         torch.cat(locs_propose_pos, dim=-1).view(-1, self.num_modes, 1, self.output_dim),
        #         dim=-2)
        #     scale_propose_pos = torch.cumsum(
        #         F.elu_(
        #             torch.cat(scales_propose_pos, dim=-1).view(-1, self.num_modes, 1, self.output_dim),
        #             alpha=1.0) +
        #         1.0, dim=-2) + 0.1
        #     if self.output_head:
        #         loc_propose_head = torch.cumsum(torch.tanh(torch.cat(locs_propose_head, dim=-1)) * math.pi, dim=-2)

        #         conc_propose_head = 1.0 / (torch.cumsum(F.elu_(torch.cat(concs_propose_head, dim=-1)) + 1.0, dim=-2) + 0.02)
        #         m = self.y_emb(torch.cat([loc_propose_pos.detach(),
        #                                 wrap_angle(loc_propose_head.detach())], dim=-1).view(-1, self.output_dim + 1))
        #     else:
        #         loc_propose_head = loc_propose_pos.new_zeros((loc_propose_pos.size(0), self.num_modes,
        #                                                     1, 1))
        #         conc_propose_head = scale_propose_pos.new_zeros((scale_propose_pos.size(0), self.num_modes,
        #                                                         1, 1))
        #         m = self.y_emb(loc_propose_pos.detach().view(-1, self.output_dim))
        #     # m = m.reshape(-1, self.num_future_steps, self.hidden_dim).transpose(0, 1)
        #     # m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0)
            
        #     for i in range(self.num_layers):
        #         m = self.t2m_refine_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
        #         m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        #         m = self.pl2m_refine_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
        #         m = self.a2m_refine_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
        #         m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        #     m = self.m2m_refine_attn_layer(m, None, edge_index_m2m)
        #     m = m.reshape(-1, self.num_modes, self.hidden_dim)
        #     loc_refine_pos = self.to_loc_refine_pos(m).view(num_agents, self.num_modes, 1, self.output_dim)
        #     loc_refine_pos = loc_refine_pos + loc_propose_pos.detach()
        #     scale_refine_pos = F.elu_(
        #         self.to_scale_refine_pos(m).view(num_agents, self.num_modes, 1, self.output_dim),
        #         alpha=1.0) + 1.0 + 0.1
        #     if self.output_head:
        #         loc_refine_head = torch.tanh(self.to_loc_refine_head(m).unsqueeze(-1)).view(-1, self.num_modes, 1, 1) * math.pi
        #         loc_refine_head = loc_refine_head + loc_propose_head.detach()
        #         conc_refine_head = 1.0 / (F.elu_(self.to_conc_refine_head(m).unsqueeze(-1)) + 1.0 + 0.02).view(-1, self.num_modes, 1, 1)
        #     else:
        #         loc_refine_head = loc_refine_pos.new_zeros((loc_refine_pos.size(0), self.num_modes, 1,
        #                                                     1))
        #         conc_refine_head = scale_refine_pos.new_zeros((scale_refine_pos.size(0), self.num_modes,
        #                                                     1, 1))
        #     # import pdb; pdb.set_trace()
        #     pi = self.to_pi(m).squeeze(-1).reshape(num_agents, self.num_modes, 1)

        #     return {
        #         'loc_propose_pos': loc_propose_pos,
        #         'scale_propose_pos': scale_propose_pos,
        #         'loc_propose_head': loc_propose_head,
        #         'conc_propose_head': conc_propose_head,
        #         'loc_refine_pos': loc_refine_pos,
        #         'scale_refine_pos': scale_refine_pos,
        #         'loc_refine_head': loc_refine_head,
        #         'conc_refine_head': conc_refine_head,
        #         'pi': pi,
        #     }
        
        # num_total_steps = 110
        num_agents, num_total_steps = data['agent']['position'].shape[:2]
        
        pos_m = data['agent']['position'][:, :, :self.input_dim].reshape(-1, self.input_dim) # [N, 2] -> [N*110, 2]
        head_m = data['agent']['heading'][:, :].reshape(-1) # [N] -> [N*110]
        head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1) # [N, 2] -> [N, 110, 2]
        total_num_nodes = pos_m.shape[0]

        # Time to m
        x_t = scene_enc['x_a'].reshape(-1, self.hidden_dim) # -> [N*110, hidden]
        x_pl = scene_enc['x_pl'][:, :].repeat(self.num_modes, 1, 1).reshape(-1, self.hidden_dim) # [N*num_modes, hiddem] -> [N*num_modes*110, hiddem]
        x_a = scene_enc['x_a'][:, :].repeat(self.num_modes, 1, 1).reshape(-1, self.hidden_dim) # [N*num_modes, 128] (last position) -> [N*num_modes*110, 128]
        m = self.mode_emb.weight.repeat(scene_enc['x_a'].size(0)*num_total_steps, 1) # [N*num_modes, 128] -> [N*num_modes*110, 128]

        mask_src = data['agent']['valid_mask'][:, :].contiguous() # [N, 50] -> [N, 110] indicating avaliable positions for all agents
        # mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False # too old sources set to false
        mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True).repeat(1, self.num_modes) # [N, num_modes] -> [N, num_modes] indicating all agents
        mask_dst = data['agent']['valid_mask'][:, :].contiguous() # all current avaliable nodes


        pos_t = data['agent']['position'][:, :, :self.input_dim].reshape(-1, self.input_dim) # [N*50, 2] -> [N*110, 2]
        head_t = data['agent']['heading'][:, :].reshape(-1) # [N*50] -> [N*110]

        mask = data['agent']['valid_mask'][:, :num_total_steps].contiguous()
        mask_t = mask.unsqueeze(2) & mask.unsqueeze(1) # shape [N, 110, 110], (i, j) is True only when both i and j are true in mask (valid for both time step i and j)
        edge_index_t2m = dense_to_sparse(mask_t)[0] # sparse indexes of mask_t, of shape [2, N1], corresponds to the indexes of true values in mask_t
        edge_index_t2m = edge_index_t2m[:, (edge_index_t2m % num_total_steps)[1] > (edge_index_t2m % num_total_steps)[0]] # only keep the time steps that j is larger than i (keeps half to avoid duplicated index)
        
        self.time_span = 10 
        edge_index_t2m = edge_index_t2m[:, edge_index_t2m[1] - edge_index_t2m[0] <= self.time_span] # ensures that edges that j - i < historical time steps
        
        
        # edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst[:, -1:].unsqueeze(1)) # [2, N1] all avaliable nodes pointing towards the current node -> all past nodes pointing towards current nodes
        rel_pos_t2m = pos_t[edge_index_t2m[0]] - pos_m[edge_index_t2m[1]] # [N1, 2] N1=number of avaliable edges from all avaliable nodes to current nodes

        rel_head_t2m = wrap_angle(head_t[edge_index_t2m[0]] - head_m[edge_index_t2m[1]]) # all avaliable headings to the current heading
        r_t2m = torch.stack(
            [torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1), 
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m[1]], nbr_vector=rel_pos_t2m[:, :2]),
             rel_head_t2m,
             (edge_index_t2m[0] % num_total_steps) - num_total_steps + 1], dim=-1)
        r_t2m = self.r_t2m_emb(continuous_inputs=r_t2m, categorical_embs=None)
        # edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1)) # [2, N1*num_modes]
        r_t2m = r_t2m.repeat_interleave(repeats=self.num_modes, dim=0) # [N1*num_modes, 128] repeated num_mode times


        # Polylines to m
        pos_pl = data['map_polygon']['position'][:, :self.input_dim] # [M1, 2]
        orient_pl = data['map_polygon']['orientation'] # [M1]

        # [2, N2] N2 = number of edges between polylines and current nodes
        # x: [N, 2]
        # y: [M1, 2]
        edge_index_pl2m = radius(
            x=pos_m[:, :2], 
            y=pos_pl[:, :2],
            r=self.pl2m_radius,
            batch_x=data['agent']['batch'].repeat_interleave(110) if isinstance(data, Batch) else None,
            batch_y=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
            max_num_neighbors=10000) # edge_index_pl2m[0] are indexes of y, edge_index_pl2m[0] are indexes of x
        # originally, there were pl pointing towards current node, now there are pl pointing towards all nodes
        # so the max neighbor limit is multiplied by 110
        edge_index_pl2m = edge_index_pl2m[:, mask_dst.reshape(-1)[edge_index_pl2m[1]]] # selecting only those whose current nodes are avaliable
        rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]] # [N2, 2] relative positions between polyline positions and current position
        rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]]) # [N2]
        r_pl2m = torch.stack( # [N2, 3]
            [torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]),
             rel_orient_pl2m], dim=-1)
        r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m, categorical_embs=None) # [N2, 128]
        edge_index_pl2m = torch.cat([edge_index_pl2m + i * edge_index_pl2m.new_tensor(
            [[data['map_polygon']['num_nodes']], [data['agent']['num_nodes']]]) for i in range(self.num_modes)], dim=1) # [2, N2*num_modes]
        r_pl2m = r_pl2m.repeat(self.num_modes, 1) # [N2*num_modes, 2]

        edge_index_a2m = radius_graph( # [2, N3] current nodes pointing towards each other
            x=pos_m[:, :2],
            r=self.a2m_radius,
            batch=data['agent']['batch'].repeat_interleave(110) if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=10000) # 110 times of original num of neighborhoods
        edge_index_a2m = edge_index_a2m[:, edge_index_a2m[1] > edge_index_a2m[0]] # this is supposed to capture more information than before: before there is only edges in the current time span, now there are edges from history to current
        # create a time mask to select from the edge_index_a2m so that only current pointing to the current
        edge_index_a2m = edge_index_a2m[:, mask_src.reshape(-1)[edge_index_a2m[0]] & mask_dst.reshape(-1)[edge_index_a2m[1]]] # [2, N3'] select edges whose both side of nodes are valid
        rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]] # [N3', 2] relative positions of current nodes
        rel_head_a2m = wrap_angle(head_m[edge_index_a2m[0]] - head_m[edge_index_a2m[1]]) # [N3']
        r_a2m = torch.stack( # [N3, 3]
            [torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2m[1]], nbr_vector=rel_pos_a2m[:, :2]),
             rel_head_a2m], dim=-1)
        r_a2m = self.r_a2m_emb(continuous_inputs=r_a2m, categorical_embs=None)
        edge_index_a2m = torch.cat(
            [edge_index_a2m + i * edge_index_a2m.new_tensor([data['agent']['num_nodes']]) for i in
             range(self.num_modes)], dim=1)
        r_a2m = r_a2m.repeat(self.num_modes, 1)

        # edge_index_m2m = dense_to_sparse(mask_dst.reshape(-1, 1).unsqueeze(2) & mask_dst.reshape(-1, 1).unsqueeze(1))[0] # edges between current avaliable nodes
        # edge_index_m2m = dense_to_sparse(mask_dst.reshape(-1, 1).unsqueeze(2) & mask_dst.reshape(-1, 1).unsqueeze(1))[0] # edges between current avaliable nodes
        # edge_index_m2m = radius_graph( # [2, N3] current nodes pointing towards each other
        #     x=pos_m[:, :2],
        #     r=100000,
        #     batch=data['agent']['batch'].repeat_interleave(110) if isinstance(data, Batch) else None,
        #     loop=False,
        #     max_num_neighbors=110*110)

        edge_index_m2m = dense_to_sparse(mask_dst.reshape(-1).unsqueeze(0).repeat(total_num_nodes, 1))[0] # edges between current avaliable nodes
        edge_index_m2m = edge_index_m2m[:, edge_index_m2m[0]//num_total_steps == edge_index_m2m[1]//num_total_steps]


        locs_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        scales_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        locs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        concs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps

        for t in range(self.num_recurrent_steps):
            for i in range(self.num_layers):
                # import pdb; pdb.set_trace()
                m = m.reshape(-1, self.hidden_dim) # [N*num_modes, hidden_dim] -> [N*num_modes*110, hidden_dim]
                m = self.t2m_propose_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m) # integrate information of other agents and all time span to m
                m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                m = self.pl2m_propose_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m) # integrate the map info into m
                m = self.a2m_propose_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m) # integrate the info of other agents in the current time to m
                m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.m2m_propose_attn_layer(m, None, edge_index_m2m) # self attention
            m = m.reshape(-1, self.num_modes, self.hidden_dim) # [N, num_modes, 128]
            locs_propose_pos[t] = self.to_loc_propose_pos(m).reshape(num_agents, num_total_steps, 2) # [N*110, num_modes, 2]
            scales_propose_pos[t] = self.to_scale_propose_pos(m).reshape(num_agents, num_total_steps, 2)
            if self.output_head:
                locs_propose_head[t] = self.to_loc_propose_head(m).view(-1, self.num_modes, self.num_future_steps, 1)
                concs_propose_head[t] = self.to_conc_propose_head(m).view(-1, self.num_modes, self.num_future_steps, 1)
        loc_propose_pos = torch.cumsum(
            torch.cat(locs_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
            dim=-2)
        scale_propose_pos = torch.cumsum(
            F.elu_(
                torch.cat(scales_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
                alpha=1.0) +
            1.0,
            dim=-2) + 0.1
        if self.output_head:
            loc_propose_head = torch.cumsum(torch.tanh(torch.cat(locs_propose_head, dim=-1)) * math.pi, dim=-2)
                                                    
            conc_propose_head = 1.0 / (torch.cumsum(F.elu_(torch.cat(concs_propose_head, dim=-1)) + 1.0, dim=-2) + 0.02)
            m = self.y_emb(torch.cat([loc_propose_pos.detach(),
                                      wrap_angle(loc_propose_head.detach())], dim=-1).view(-1, self.output_dim + 1))
        else:
            loc_propose_head = loc_propose_pos.new_zeros((loc_propose_pos.size(0), self.num_modes,
                                                          self.num_future_steps, 1))
            conc_propose_head = scale_propose_pos.new_zeros((scale_propose_pos.size(0), self.num_modes,
                                                             self.num_future_steps, 1))
            m = self.y_emb(loc_propose_pos.detach().view(-1, self.output_dim))
        # m = m.reshape(-1, self.num_future_steps, self.hidden_dim).transpose(0, 1)
        # m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0)
        
        for i in range(self.num_layers):
            m = self.t2m_refine_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
            m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.pl2m_refine_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
            m = self.a2m_refine_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
            m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        m = self.m2m_refine_attn_layer(m, None, edge_index_m2m)
        m = m.reshape(-1, self.num_modes, self.hidden_dim)
        loc_refine_pos = self.to_loc_refine_pos(m).view(num_agents, self.num_modes, self.num_future_steps, self.output_dim)
        loc_refine_pos = loc_refine_pos + loc_propose_pos.detach()
        scale_refine_pos = F.elu_(
            self.to_scale_refine_pos(m).view(num_agents, self.num_modes, self.num_future_steps, self.output_dim),
            alpha=1.0) + 1.0 + 0.1
        if self.output_head:
            loc_refine_head = torch.tanh(self.to_loc_refine_head(m).unsqueeze(-1)).view(-1, self.num_modes, self.num_future_steps, 1) * math.pi
            loc_refine_head = loc_refine_head + loc_propose_head.detach()
            conc_refine_head = 1.0 / (F.elu_(self.to_conc_refine_head(m).unsqueeze(-1)) + 1.0 + 0.02).view(-1, self.num_modes, self.num_future_steps, 1)
        else:
            loc_refine_head = loc_refine_pos.new_zeros((loc_refine_pos.size(0), self.num_modes, self.num_future_steps,
                                                        1))
            conc_refine_head = scale_refine_pos.new_zeros((scale_refine_pos.size(0), self.num_modes,
                                                           self.num_future_steps, 1))
        # import pdb; pdb.set_trace()
        pi = self.to_pi(m).squeeze(-1).reshape(num_agents, self.num_modes, num_total_steps)
        
        # import pdb; pdb.set_trace()
        return {
            'loc_propose_pos': loc_propose_pos[:, :, self.num_historical_steps:],
            'scale_propose_pos': scale_propose_pos[:, :, self.num_historical_steps:],
            'loc_propose_head': loc_propose_head[:, :, self.num_historical_steps:],
            'conc_propose_head': conc_propose_head[:, :, self.num_historical_steps:],
            'loc_refine_pos': loc_refine_pos[:, :, self.num_historical_steps:],
            'scale_refine_pos': scale_refine_pos[:, :, self.num_historical_steps:],
            'loc_refine_head': loc_refine_head[:, :, self.num_historical_steps:],
            'conc_refine_head': conc_refine_head[:, :, self.num_historical_steps:],
            'pi': pi[:, :, self.num_historical_steps:],
        }
        
        # pos_m = data['agent']['position'][:, self.num_historical_steps - 1, :self.input_dim] # [N, 2] -> [N, 110, 2]
        # head_m = data['agent']['heading'][:, self.num_historical_steps - 1] # [N] -> [N, 110]
        # head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1) # [N, 2] -> [N, 110, 2]

        # x_t = scene_enc['x_a'].reshape(-1, self.hidden_dim) # -> [N*110, hidden]
        # x_pl = scene_enc['x_pl'][:, self.num_historical_steps - 1].repeat(self.num_modes, 1) # [N*num_modes, hiddem] [N*num_modes, 110, hidden]
        # x_a = scene_enc['x_a'][:, -1].repeat(self.num_modes, 1) # [N*num_modes, 1] (last position) -> [N*num_modes, 110]
        # m = self.mode_emb.weight.repeat(scene_enc['x_a'].size(0), 1) # [N*num_modes, 128]

        # mask_src = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous() # [N, 50] -> [N, 110] indicating avaliable positions for all agents
        # mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        # mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True).repeat(1, self.num_modes) # [N, num_modes] -> [N, num_modes] indicating all agents

        # pos_t = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim) # [N*50, 2] -> [N*110, 2]
        # head_t = data['agent']['heading'][:, :self.num_historical_steps].reshape(-1) # [N*50] -> [N*110]
        # edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst[:, -1:].unsqueeze(1)) # [2, N1] all avaliable nodes pointing towards the current node
        # rel_pos_t2m = pos_t[edge_index_t2m[0]] - pos_m[edge_index_t2m[1]] # [N1, 2] N1=number of avaliable edges from all avaliable nodes to current nodes

        # rel_head_t2m = wrap_angle(head_t[edge_index_t2m[0]] - head_m[edge_index_t2m[1]]) # all avaliable headings to the current heading
        # r_t2m = torch.stack(
        #     [torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1), 
        #      angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m[1]], nbr_vector=rel_pos_t2m[:, :2]),
        #      rel_head_t2m,
        #      (edge_index_t2m[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)
        # r_t2m = self.r_t2m_emb(continuous_inputs=r_t2m, categorical_embs=None)
        # edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1)) # [2, N1*num_modes]
        # r_t2m = r_t2m.repeat_interleave(repeats=self.num_modes, dim=0) # [N1*num_modes, 128] repeated num_mode times

        # pos_pl = data['map_polygon']['position'][:, :self.input_dim] # [M1, 2]
        # orient_pl = data['map_polygon']['orientation'] # [M1]
        # edge_index_pl2m = radius( # [2, N2] N2 = number of edges between polylines and current nodes
        #     x=pos_m[:, :2], # [N, 2]
        #     y=pos_pl[:, :2], # [M1, 2]
        #     r=self.pl2m_radius,
        #     batch_x=data['agent']['batch'] if isinstance(data, Batch) else None,
        #     batch_y=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
        #     max_num_neighbors=300) # edge_index_pl2m[0] are indexes of y, edge_index_pl2m[0] are indexes of x
        # edge_index_pl2m = edge_index_pl2m[:, mask_dst[edge_index_pl2m[1], 0]] # selecting only those whose current nodes are avaliable
        # rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]] # [N2, 2] relative positions between polyline positions and current position
        # rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]]) # [N2]
        # r_pl2m = torch.stack( # [N2, 3]
        #     [torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
        #      angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]),
        #      rel_orient_pl2m], dim=-1)
        # r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m, categorical_embs=None) # [N2, 128]
        # edge_index_pl2m = torch.cat([edge_index_pl2m + i * edge_index_pl2m.new_tensor(
        #     [[data['map_polygon']['num_nodes']], [data['agent']['num_nodes']]]) for i in range(self.num_modes)], dim=1) # [2, N2*num_modes]
        # r_pl2m = r_pl2m.repeat(self.num_modes, 1) # [N2*num_modes, 2]

        # edge_index_a2m = radius_graph( # [2, N3] current nodes pointing towards each other
        #     x=pos_m[:, :2],
        #     r=self.a2m_radius,
        #     batch=data['agent']['batch'] if isinstance(data, Batch) else None,
        #     loop=False,
        #     max_num_neighbors=300)
        # # create a time mask to select from the edge_index_a2m so that only current pointing to the current
        # edge_index_a2m = edge_index_a2m[:, mask_src[:, -1][edge_index_a2m[0]] & mask_dst[edge_index_a2m[1], 0]] # [2, N3'] select edges whose both side of nodes are valid
        # rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]] # [N3', 2] relative positions of current nodes
        # rel_head_a2m = wrap_angle(head_m[edge_index_a2m[0]] - head_m[edge_index_a2m[1]]) # [N3']
        # r_a2m = torch.stack( # [N3, 3]
        #     [torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
        #      angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2m[1]], nbr_vector=rel_pos_a2m[:, :2]),
        #      rel_head_a2m], dim=-1)
        # r_a2m = self.r_a2m_emb(continuous_inputs=r_a2m, categorical_embs=None)
        # edge_index_a2m = torch.cat(
        #     [edge_index_a2m + i * edge_index_a2m.new_tensor([data['agent']['num_nodes']]) for i in
        #      range(self.num_modes)], dim=1)
        # r_a2m = r_a2m.repeat(self.num_modes, 1)

        # edge_index_m2m = dense_to_sparse(mask_dst.unsqueeze(2) & mask_dst.unsqueeze(1))[0] # edges between current avaliable nodes

        # locs_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        # scales_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        # locs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        # concs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps

        # for t in range(self.num_recurrent_steps):
        #     for i in range(self.num_layers):
        #         m = m.reshape(-1, self.hidden_dim) # [N*num_modes, hidden_dim] -> [N*num_modes*110, hidden_dim]
        #         m = self.t2m_propose_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m) # integrate information of other agents and all time span to m
        #         m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        #         m = self.pl2m_propose_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m) # integrate the map info into m
        #         m = self.a2m_propose_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m) # integrate the info of other agents in the current time to m
        #         m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        #     m = self.m2m_propose_attn_layer(m, None, edge_index_m2m) # self attention
        #     m = m.reshape(-1, self.num_modes, self.hidden_dim) # [N, num_modes, 128]
        #     locs_propose_pos[t] = self.to_loc_propose_pos(m) # [N, num_modes, ]
        #     scales_propose_pos[t] = self.to_scale_propose_pos(m)
        #     if self.output_head:
        #         locs_propose_head[t] = self.to_loc_propose_head(m)
        #         concs_propose_head[t] = self.to_conc_propose_head(m)
        # loc_propose_pos = torch.cumsum(
        #     torch.cat(locs_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
        #     dim=-2)
        # scale_propose_pos = torch.cumsum(
        #     F.elu_(
        #         torch.cat(scales_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
        #         alpha=1.0) +
        #     1.0,
        #     dim=-2) + 0.1
        # if self.output_head:
        #     loc_propose_head = torch.cumsum(torch.tanh(torch.cat(locs_propose_head, dim=-1).unsqueeze(-1)) * math.pi,
        #                                     dim=-2)
        #     conc_propose_head = 1.0 / (torch.cumsum(F.elu_(torch.cat(concs_propose_head, dim=-1).unsqueeze(-1)) + 1.0,
        #                                             dim=-2) + 0.02)
        #     m = self.y_emb(torch.cat([loc_propose_pos.detach(),
        #                               wrap_angle(loc_propose_head.detach())], dim=-1).view(-1, self.output_dim + 1))
        # else:
        #     loc_propose_head = loc_propose_pos.new_zeros((loc_propose_pos.size(0), self.num_modes,
        #                                                   self.num_future_steps, 1))
        #     conc_propose_head = scale_propose_pos.new_zeros((scale_propose_pos.size(0), self.num_modes,
        #                                                      self.num_future_steps, 1))
        #     m = self.y_emb(loc_propose_pos.detach().view(-1, self.output_dim))
        # m = m.reshape(-1, self.num_future_steps, self.hidden_dim).transpose(0, 1)
        # m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0)
        # for i in range(self.num_layers):
        #     m = self.t2m_refine_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
        #     m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        #     m = self.pl2m_refine_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
        #     m = self.a2m_refine_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
        #     m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        # m = self.m2m_refine_attn_layer(m, None, edge_index_m2m)
        # m = m.reshape(-1, self.num_modes, self.hidden_dim)
        # loc_refine_pos = self.to_loc_refine_pos(m).view(-1, self.num_modes, self.num_future_steps, self.output_dim)
        # loc_refine_pos = loc_refine_pos + loc_propose_pos.detach()
        # scale_refine_pos = F.elu_(
        #     self.to_scale_refine_pos(m).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
        #     alpha=1.0) + 1.0 + 0.1
        # if self.output_head:
        #     loc_refine_head = torch.tanh(self.to_loc_refine_head(m).unsqueeze(-1)) * math.pi
        #     loc_refine_head = loc_refine_head + loc_propose_head.detach()
        #     conc_refine_head = 1.0 / (F.elu_(self.to_conc_refine_head(m).unsqueeze(-1)) + 1.0 + 0.02)
        # else:
        #     loc_refine_head = loc_refine_pos.new_zeros((loc_refine_pos.size(0), self.num_modes, self.num_future_steps,
        #                                                 1))
        #     conc_refine_head = scale_refine_pos.new_zeros((scale_refine_pos.size(0), self.num_modes,
        #                                                    self.num_future_steps, 1))
        # pi = self.to_pi(m).squeeze(-1)

        # return {
        #     'loc_propose_pos': loc_propose_pos,
        #     'scale_propose_pos': scale_propose_pos,
        #     'loc_propose_head': loc_propose_head,
        #     'conc_propose_head': conc_propose_head,
        #     'loc_refine_pos': loc_refine_pos,
        #     'scale_refine_pos': scale_refine_pos,
        #     'loc_refine_head': loc_refine_head,
        #     'conc_refine_head': conc_refine_head,
        #     'pi': pi,
        # }