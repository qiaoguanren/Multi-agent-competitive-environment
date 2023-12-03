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
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import ArgoverseV2DataModule
from predictors import QCNet

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)


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
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import ArgoverseV2DataModule
from predictors import QCNet

import os
from pytorch_lightning.loggers import TensorBoardLogger
import yaml

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_qcnet.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        try:
            config_args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    checkpoint_path = config_args.get('checkpoint_path', None)
    version_folder = config_args.get('version_folder', None)
    if version_folder != None:
        # Find the latest checkpoint in the version folder
        print('resumed from version: ', version_folder)
        checkpoint_dir = os.path.join(version_folder, "checkpoints")
        list_of_files = [f for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f))]
        list_of_files = sorted(list_of_files, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
        checkpoint_path = os.path.join(checkpoint_dir, list_of_files[-1])  # Last file is the latest
        version = os.path.basename(version_folder)
        logger = TensorBoardLogger("lightning_logs", name="", version=version)
        model = QCNet.load_from_checkpoint(checkpoint_path)


    elif checkpoint_path and checkpoint_path != 'None':
        model = QCNet.load_from_checkpoint(checkpoint_path)
    else:
        model = QCNet(**config_args['QCNet'])
        
        
    checkpoint_path = config_args.get('checkpoint_path', None)
    if checkpoint_path and checkpoint_path != 'None':
        model = QCNet.load_from_checkpoint(checkpoint_path)
    else:
        model = QCNet(**config_args['QCNet'])

    model.lr = config_args['QCNet']['lr']
    model.evaluate_type = config_args['QCNet']['evaluate_type']
    
    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
    }[config_args['QCNet']['dataset']](**config_args)
    model_checkpoint = ModelCheckpoint(monitor='val_minADE', save_top_k=5, mode='min')
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(accelerator=config_args['accelerator'], devices=config_args['devices'],
                         strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=config_args['max_epochs'],
                         precision="16")
    # trainer = pl.Trainer(accelerator=config_args['accelerator'], devices=config_args['devices'],
    #                      strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
    #                      callbacks=[model_checkpoint, lr_monitor], max_epochs=config_args['max_epochs'],
    #                      val_check_interval=1, precision="16")
    
    trainer.fit(model, datamodule)
