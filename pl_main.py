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
from pl_trainSides import sidesModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from datetime import datetime
import pprint

def get_timestamp():
    dt = datetime.now()
    return  f"{dt.month:02d}-{dt.day:02d}-{dt.hour:02d}-{dt.minute:02d}"

if __name__ == "__main__":
    opt = parse_args() 
    hparams = vars(opt)
    pprint.pprint(hparams)

    timestmap = get_timestamp()
    logfile_name = f'{opt.tgt_dir}/{opt.prefix}{timestmap}'+\
        f'_{opt.mode}_{opt.loss}_{opt.normalization}/'

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
    # data_module.setup()

    writer = f'{opt.prefix}'

    logger =  WandbLogger(project='SDAN',
                            name=writer,
                            save_dir=opt.tgt_dir)    
    
    pl.seed_everything(42, workers=True)
    model = sidesModel(
        weight_sides=opt.weight_sides,
        pretrained_style=opt.pretrained_style,
        loss=opt.loss,
        burnin=opt.burnin,
        normalization=opt.normalization,
        writer=writer,
        epochs=opt.num_epochs,
        learning_rate=opt.lr,
        logfile=f'{logfile_name}/log.txt',
        num_epochs=opt.num_epochs
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
            normalization=opt.normalization,
            writer=writer,
            epochs=opt.num_epochs,
            learning_rate=opt.lr,
            logfile=f'{logfile_name}/log.txt',
            max_iters=opt.num_epochs
        )  
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = [lr_monitor, ModelCheckpoint(
        monitor="Loss/val",
        mode ='min',
        dirpath=logfile_name)
        , 
        EarlyStopping(
                monitor='Loss/val',
                min_delta=0.00,
                patience=20,
                verbose=False,
                mode='min')
        ]

    trainer = pl.Trainer(
        devices=torch.cuda.device_count(),
        accelerator='gpu',
        default_root_dir=logfile_name,
        logger=logger,
        max_epochs=opt.num_epochs,
        profiler="simple",
        callbacks=checkpoint_callback,
        log_every_n_steps=5,
        # val_check_interval=0.25,
        precision=32,
    )

    trainer.fit(model, datamodule=data_module)

    trainer.test(model, data_module)