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
from itertools import chain
from itertools import compress
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from losses import MixtureNLLLoss
from losses import NLLLoss
from metrics import Brier
from metrics import MR
from metrics import minADE
from metrics import minAHE
from metrics import minFDE
from metrics import minFHE
from modules import QCNetDecoder
from modules import QCNetEncoder
from predictors.qcnet import QCNet
from utils import wrap_angle
import torch.nn.functional as F

import random

try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object


class AntoQCNet(QCNet):

    def forward(self, data: HeteroData):
        scene_enc = self.encoder(data)
        pred = self.decoder(data, scene_enc)
        return pred

    def validation_step(self,
                        data,
                        batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        pred = self(data)
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)

        def get_auto_pred(data, loc_refine_pos, scale_refine_pos, offset):
            
            origin = data["agent"]["position"][:, self.num_historical_steps - 1]
            theta = data["agent"]["heading"][:, self.num_historical_steps - 1]
            cos, sin = theta.cos(), theta.sin()
            rot_mat = theta.new_zeros(data["agent"]["num_nodes"], 2, 2)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = -sin
            rot_mat[:, 1, 0] = sin
            rot_mat[:, 1, 1] = cos

            # auto_index = data['agent']['valid_mask'][:,self.num_historical_steps]
            data["agent"]["valid_mask"] = (
                torch.cat(
                    (
                        data["agent"]["valid_mask"][..., offset:],
                        torch.zeros(data["agent"]["valid_mask"].shape[:-1] + (offset,)).cuda(),
                    ),
                    dim=-1,
                )
            ).bool()
            data["agent"]["valid_mask"][:, 0] = False

            new_position = torch.matmul(
                loc_refine_pos[..., :2], rot_mat.swapaxes(-1, -2)
            ) + origin[:, :2].unsqueeze(1)

            input_position = torch.zeros_like(data["agent"]["position"])
            input_position[:, :-offset] = data["agent"]["position"][:, offset:]
            input_position[
                :, self.num_historical_steps - offset : self.num_historical_steps, :2
            ] = new_position[:, :offset]

            input_heading = torch.zeros_like(data["agent"]["heading"])
            input_heading[:, :-offset] = data["agent"]["heading"][:, offset:]
            input_heading[
                :, self.num_historical_steps - offset : self.num_historical_steps
            ] = wrap_angle(scale_refine_pos[..., :offset, 1] + theta.unsqueeze(-1))

            input_v = torch.zeros_like(data["agent"]["velocity"])
            input_v[:, :-offset] = data["agent"]["velocity"][:, offset:]
            input_v[:, self.num_historical_steps - offset : self.num_historical_steps, :2] = (
                new_position[:, 1:] - new_position[:, :-1]
            )[:, :offset] / 0.1

            data["agent"]["position"] = input_position
            data["agent"]["heading"] = input_heading
            data["agent"]["velocity"] = input_v

            auto_pred = self(data)

            new_origin = data["agent"]["position"][:, self.num_historical_steps - 1]
            new_theta = data["agent"]["heading"][:, self.num_historical_steps - 1]
            new_cos, new_sin = new_theta.cos(), new_theta.sin()
            new_rot_mat = new_theta.new_zeros(data["agent"]["num_nodes"], 2, 2)
            new_rot_mat[:, 0, 0] = new_cos
            new_rot_mat[:, 0, 1] = -new_sin
            new_rot_mat[:, 1, 0] = new_sin
            new_rot_mat[:, 1, 1] = new_cos

            new_trans_position_propose = torch.einsum(
                "bijk,bkn->bijn",
                auto_pred["loc_propose_pos"][..., : self.output_dim],
                new_rot_mat.swapaxes(-1, -2),
            ) + new_origin[:, :2].unsqueeze(1).unsqueeze(1)
            auto_pred["loc_propose_pos"][..., : self.output_dim] = torch.einsum(
                "bijk,bkn->bijn",
                new_trans_position_propose - origin[:, :2].unsqueeze(1).unsqueeze(1),
                rot_mat,
            )
            auto_pred["scale_propose_pos"][..., self.output_dim - 1] = wrap_angle(
                auto_pred["scale_propose_pos"][..., self.output_dim - 1]
                + new_theta.unsqueeze(-1).unsqueeze(-1)
                - theta.unsqueeze(-1).unsqueeze(-1)
            )

            new_trans_position_refine = torch.einsum(
                "bijk,bkn->bijn",
                auto_pred["loc_refine_pos"][..., : self.output_dim],
                new_rot_mat.swapaxes(-1, -2),
            ) + new_origin[:, :2].unsqueeze(1).unsqueeze(1)
            auto_pred["loc_refine_pos"][..., : self.output_dim] = torch.einsum(
                "bijk,bkn->bijn",
                new_trans_position_refine - origin[:, :2].unsqueeze(1).unsqueeze(1),
                rot_mat,
            )
            auto_pred["scale_refine_pos"][..., self.output_dim - 1] = wrap_angle(
                auto_pred["scale_refine_pos"][..., self.output_dim - 1]
                + new_theta.unsqueeze(-1).unsqueeze(-1)
                - theta.unsqueeze(-1).unsqueeze(-1)
            )

            if self.output_head:
                auto_traj_propose = torch.cat(
                    [
                        auto_pred["loc_propose_pos"][..., : self.output_dim],
                        auto_pred["loc_propose_head"],
                        auto_pred["scale_propose_pos"][..., : self.output_dim],
                        auto_pred["conc_propose_head"],
                    ],
                    dim=-1,
                )
                auto_traj_refine = torch.cat(
                    [
                        auto_pred["loc_refine_pos"][..., : self.output_dim],
                        auto_pred["loc_refine_head"],
                        auto_pred["scale_refine_pos"][..., : self.output_dim],
                        auto_pred["conc_refine_head"],
                    ],
                    dim=-1,
                )
            else:
                auto_traj_propose = torch.cat([auto_pred['loc_propose_pos'][..., :self.output_dim],
                                        auto_pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
                auto_traj_refine = torch.cat([auto_pred['loc_refine_pos'][..., :self.output_dim],
                                        auto_pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)

            new_data = data
            return (
                new_data,
                auto_pred,
                auto_traj_refine,
                auto_traj_propose,
                (new_trans_position_propose, new_trans_position_refine),
            )

        offset=5
        pi = pred['pi']
        new_data = data.clone()
        auto_pred = pred
        auto_traj_refine = traj_refine
        auto_traj_propose = traj_propose
        new_pi = pi.clone()

        for i in range(self.num_future_steps,5):
          gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
          l2_norm = (torch.norm(auto_traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
          best_mode = l2_norm.argmin(dim=-1)
          traj_propose[:,:,i:i+offset]=auto_traj_propose[torch.arange(auto_traj_propose.size(0)),best_mode,:offset].unsqueeze(1)
          traj_refine[:,:,i:i+offset]=auto_traj_refine[torch.arange(auto_traj_refine.size(0)),best_mode,:offset].unsqueeze(1)
          # traj_propose[:,:,i+1:]=auto_traj_propose[:,:,:-(i+1)]
          # traj_refine[:,:,i+1:]=auto_traj_refine[:,:,:-(i+1)]

          new_data, auto_pred, auto_traj_refine, auto_traj_propose, new_position = get_auto_pred(
              new_data, auto_pred["loc_refine_pos"][torch.arange(auto_traj_refine.size(0)),best_mode], auto_pred["scale_refine_pos"][torch.arange(auto_traj_refine.size(0)),best_mode], offset
          )

        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        # best_mode = torch.randint(low=0, high=6, size=(traj_propose.size(0),))
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]

        mse_loss = F.mse_loss(traj_refine_best[..., :self.output_dim], gt[..., :self.output_dim])
        print('mse_loss\t',mse_loss)

        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        # cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
        #                          target=gt[:, -1:, :self.output_dim + self.output_head],
        #                          prob=pi,
        #                          mask=reg_mask[:, -1:]) * cls_mask
        # cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        # self.log('val_cls_loss', cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        valid_mask_eval = reg_mask[eval_mask]
        traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
        if not self.output_head:
            traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                     traj_eval[..., :2]], dim=-2)
            motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
            head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
            traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)
        gt_eval = gt[eval_mask]

        self.Brier.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                          valid_mask=valid_mask_eval)
        self.minADE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.minAHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.minFDE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.minFHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.MR.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                       valid_mask=valid_mask_eval)
        self.log('val_Brier', self.Brier, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minAHE', self.minAHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFHE', self.minFHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
