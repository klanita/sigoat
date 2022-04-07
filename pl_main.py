# import sys
# sys.path.append("/home/anna/dlbirhoui/")
# sys.path.append("/home/anna/dlbirhoui/fadern/")
import os
import math
import numpy as np
import torch
from options import parse_args
import pl_data
from datetime import date
# from utils import *
import pytorch_lightning as pl

import random
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

from sklearn.model_selection import train_test_split
import pprint
import json
from clearml import Task
from pl_trainSides import sidesModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    opt = parse_args() 

    logfile_name = f'{opt.tgt_dir}/{opt.prefix}{str(date.today())}'+\
        f'_{opt.mode}/'

    if not os.path.exists(logfile_name):
        os.makedirs(logfile_name)
    print('Results will be saved to:\t', logfile_name, '\n')

    with open(f'{logfile_name}/commandline_args.txt', 'w') as f:
        print(f'Parameters of the model are in: ',
            f'{logfile_name}/commandline_args.txt')
        json.dump(opt.__dict__, f, indent=2)

    msg = f'\nStart training {opt.mode.upper()} network.'
    print(msg)
    print(''.join(['=']*len(msg)))

    data_module = pl_data.OADataModule(opt)
    data_module.setup()

    writer = f'{opt.prefix}'

    task = Task.init(
            project_name='DeepTagSpectr2Seq',
            task_name=writer,
            reuse_last_task_id=False)

    pl.seed_everything(42, workers=True)
    model = sidesModel(
        weight_sides=opt.weight_sides,
        pretrained_style=opt.pretrained_style,
        loss=opt.loss,
        burnin=opt.burnin,

        writer=writer,
        epochs=opt.num_epochs,
        learning_rate=opt.lr,
        logfile=f'{logfile_name}/log.txt',
        max_iters=len(data_module.train_dataloader())*opt.num_epochs
        )  
    
    model.hparams.epochs = opt.num_epochs
    model.hparams.learning_rate = opt.lr
    
    if not (opt.pretrained is None):
            model = model.load_from_checkpoint(
            checkpoint_path=opt.pretrained,
            weight_sides=opt.weight_sides,
            pretrained_style=opt.pretrained_style,
            loss=opt.loss,
            burnin=opt.burnin,

            writer=writer,
            epochs=opt.num_epochs,
            learning_rate=opt.lr,
            logfile=f'{logfile_name}/log.txt',
            max_iters=len(data_module.train_dataloader())*opt.num_epochs
        )  
    
    checkpoint_callback = ModelCheckpoint(
        monitor="Loss/val",
        mode ='min',
        dirpath=logfile_name)

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        auto_lr_find=True,
        default_root_dir=logfile_name,
        max_epochs=opt.num_epochs,
        profiler="simple",
        callbacks=[checkpoint_callback],
        checkpoint_callback=True,
        log_every_n_steps=1,
        # val_check_interval=0.25,
        precision=16,
        strategy='dp',
        num_processes=None,
        accelerator=None,
    )

    trainer.fit(model, datamodule=data_module)