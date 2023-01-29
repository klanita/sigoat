import argparse
import torch

from dataLoader import DataModule
from model import UNet

from datetime import date
import os
import json
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import WandbLogger
import wandb


def parse_args():
    parser = argparse.ArgumentParser(
        description='Unet benchmark.')    

    # parser.add_argument('--file_in', type=str,
    #     default='/home/anna/data_19Nov2022/old_syn')
    
    parser.add_argument('--file_in', type=str,
        default='/home/anna/ResultsSignalDA/GT_syn/old_syn')
    # parser.add_argument('--file_out', type=str,
    #     default='/home/anna/data_19Nov2022/old_syn')

    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    
    parser.add_argument('--test_prefix', type=str, default="syn")
    
    # parameters for training
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=0.001)    
    parser.add_argument('--burnin', type=int, default=100)
    parser.add_argument('--n_steps', type=int, default=5)
    
    parser.add_argument('--bilinear', type=int, default=1)
    
    parser.add_argument('--scheduler', type=str, default='COSINE')
    parser.add_argument('--cosine_warmup_epochs', 
            help="Number of warmup steps for COSINE (yes, STEPS eventhough it's called epochs. Default: 10'000.",
            type=int, default=10000)
    parser.add_argument('--steplr_factor', help="Reducting factor for Step LR, Default : 0.1.",
            type=float, default=0.1)
    parser.add_argument('--steplr_patience', help="Patience for STEPLR, Default : 5.",
        type=int, default=5)
        
    parser.add_argument('--norm_layer', type=str, default='BN')
    parser.add_argument('--loss_name', type=str, default='l1', 
            help="Loss. Default: l2.", choices=['l1', 'l2'])

    parser.add_argument('--tgt_dir', type=str, 
        default="/home/anna/ResultsSignalDA/UNet-BM/",
        help="Target directory to save the model.")
    parser.add_argument('--prefix', type=str, default="maxpool")
    
    # parameters for the loss
    parser.add_argument('--loss', type=str, default="l1")
    parser.add_argument('--num_workers', type=int, default=16,\
        help='Number of workers for data loader.')
        
    parser.add_argument('--weight_decay', type=float, default=1e-5,\
        help='Weight decay.')
    
    parser.add_argument('--ckpt', help='Path to checkpoint. Default: None.', 
            default=None)

    args = parser.parse_args()

    args.bilinear = bool(args.bilinear)
    
    return args


if __name__ == '__main__':
    opt = parse_args()
    
    logfile_name = f'{opt.tgt_dir}/{opt.prefix}_{opt.loss_name}_{opt.norm_layer}_{str(date.today())}'

    if not os.path.exists(logfile_name):
        os.makedirs(logfile_name)
    print('Results will be saved to:\t', logfile_name, '\n')

    with open(f'{logfile_name}/commandline_args.txt', 'w') as f:
        print(f'Parameters of the model are in: ',
            f'{logfile_name}/commandline_args.txt')
        json.dump(opt.__dict__, f, indent=2)      
    
    params = vars(opt)
    gpu = torch.cuda.device_count()
    params['batch_size'] *= gpu
    params['task_name'] = f"{opt.prefix}_{opt.loss_name}_{opt.norm_layer}_{date.today()}"
    params['logfile_name'] = logfile_name
    
    print(gpu, params)
    
    data_module = DataModule(opt)
    
    pl.seed_everything(42, workers=True)
    model = UNet(**params)  
    
    model.hparams.epochs = opt.num_epochs
    model.hparams.learning_rate = opt.lr
    
    checkpoint_callback = ModelCheckpoint(
        monitor="Loss/val",
        mode ='min',
        dirpath=logfile_name)
    
    if not (params['ckpt'] is None):
        model.load_state_dict(torch.load(params['ckpt'])['state_dict'], strict=True)
        print('Loaded model from ckpt: ', params['ckpt'])
        
    logger =  WandbLogger(project='SDAN',
                            name=params['task_name'],
                            save_dir=opt.tgt_dir)
    
    callbacks=[checkpoint_callback]
    early_stop_callback = EarlyStopping(
                monitor='Loss/val',
                min_delta=0.00,
                patience=10,
                verbose=False,
                mode='min'
            )
    callbacks.append(early_stop_callback)
            
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=torch.cuda.device_count(),
        auto_lr_find=False,
        default_root_dir=logfile_name,
        max_epochs=opt.num_epochs,
        profiler="simple",
        callbacks=callbacks,
        logger=logger,
        # checkpoint_callback=True,
        log_every_n_steps=200,
        val_check_interval=0.5,
        # precision=16,
        strategy='dp',
        num_processes=None,
    )
    
    if opt.mode == 'train':
        trainer.fit(model, datamodule=data_module)
    
    model.eval()
    results = trainer.test(model, data_module)[0]
    print(results)
    
    wandb.finish()