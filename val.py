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
from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from predictors import QCNet
from transforms import TargetBuilder
import yaml

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    # parser.add_argument('--model', type=str, required=True)
    # parser.add_argument('--root', type=str, required=True)
    # parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--num_workers', type=int, default=8)
    # parser.add_argument('--pin_memory', type=bool, default=True)
    # parser.add_argument('--persistent_workers', type=bool, default=True)
    # parser.add_argument('--accelerator', type=str, default='auto')
    # parser.add_argument('--devices', type=int, default=1)
    # parser.add_argument('--ckpt_path', type=str, required=True)
    # args = parser.parse_args()
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/val_qcnet.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        try:
            config_args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    checkpoint_path = config_args.get('checkpoint_path', None)
    model = {
        'QCNet': QCNet,
    }[config_args['model']].load_from_checkpoint(checkpoint_path)
    
    model.evaluate_type = config_args['QCNet']['evaluate_type']
    model.num_historical_steps = config_args['QCNet']['num_historical_steps']
    model.num_future_steps = config_args['QCNet']['num_future_steps']
    
    
    
    val_dataset = {
        'argoverse_v2': ArgoverseV2Dataset,
    }[model.dataset](root=config_args['root'], split='val',
                     transform=TargetBuilder(model.num_historical_steps, model.num_future_steps, teacher_forcing=config_args['teacher_forcing']))
    
    dataloader = DataLoader(val_dataset, batch_size=config_args['batch_size'], shuffle=False, num_workers=config_args['num_workers'],
                            pin_memory=config_args['pin_memory'], persistent_workers=config_args['persistent_workers'])
    
    trainer = pl.Trainer(accelerator=config_args['accelerator'], devices=config_args['devices'], strategy='ddp')
    trainer.validate(model, dataloader)

    
#     model = {
#         'QCNet': QCNet,
#     }[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path)
#     val_dataset = {
#         'argoverse_v2': ArgoverseV2Dataset,
#     }[model.dataset](root=args.root, split='val',
#                      transform=TargetBuilder(model.num_historical_steps, model.num_future_steps))
#     dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
#                             pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
#     trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp')
#     trainer.validate(model, dataloader)
