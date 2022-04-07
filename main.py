# import sys
# sys.path.append("/home/anna/dlbirhoui/")
# sys.path.append("/home/anna/dlbirhoui/fadern/")
import os
import math
import numpy as np
import torch
from options import parse_args
# from torch.utils.tensorboard import SummaryWriter

from dataLoader import UnpairedDataset, UnpairedDatasetImages
from datetime import date
# from utils import *

import random
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

from sklearn.model_selection import train_test_split
import pprint
import json

if __name__ == "__main__":
    opt = parse_args()

    # Indexes of the images we use for validation are currently hardcoded
    if opt.test:
        idx_test = []
    else:
        idx_test = [0, 142, 285, 428, 570, 713, 856, 998, 1141, 1284, 1426,
        1569, 1712, 1854, 1997, 2140, 2282, 2425, 2568, 2710, 2853, 2996, 3138, 
        3281, 3424, 3566, 3709, 3851, 3994, 4137, 4280, 4422]

    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    # Load data
    if opt.mode in ['styleImages']:
        dataset = UnpairedDatasetImages(
            opt.file_in,
            file_name_real=opt.target)
    else:
        dataset = UnpairedDataset(
            opt.file_in,
            file_name_real=opt.target,
            geometry=opt.geometry
            )    

    # hold of test images which are never used for training
    idx_all = list(set(range(len(dataset))) - set(idx_test))

    # separate the leftover of the dataset into train and validation
    idx_train, idx_val = train_test_split(
        idx_all, test_size=32, random_state=42)
    val_set = torch.utils.data.dataset.Subset(
        dataset, torch.LongTensor(idx_val))
    train_set = torch.utils.data.dataset.Subset(
        dataset, torch.LongTensor(idx_train))
    print('Training Dataset length:', len(train_set), '\tdevice:', device)

    data_loader_train = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=opt.batch_size,
        shuffle=True,        
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=opt.batch_size,
        shuffle=False,
    )

    logfile_name = f'{opt.tgt_dir}/{opt.prefix}{str(date.today())}'+\
        f'_{opt.mode}/'

    if not os.path.exists(logfile_name):
        os.makedirs(logfile_name)
    print('Results will be saved to:\t', logfile_name, '\n')

    with open(f'{logfile_name}/commandline_args.txt', 'w') as f:
        print(f'Parameters of the model are in: ',
            f'{logfile_name}/commandline_args.txt')
        json.dump(opt.__dict__, f, indent=2)

    msg = f'\nStart training {opt.mode.upper()} network on {device}.'
    print(msg)
    print(''.join(['=']*len(msg)))

    # Training options
    if opt.mode == 'sidesTwo':
        # benchmark with a simple prediction network
        from trainSidesRegression import TrainerReconstruction
        sides = TrainerReconstruction(opt, logfile_name, device=device)
        sides.training(data_loader_train, data_loader_val,\
            num_epochs=opt.num_epochs, loss=opt.loss)   
    
    elif opt.mode == 'sidesAE':
        # reconstruct sides with domain adaptation
        from trainSidesDA import TrainerSides
        sides = TrainerSides(opt, logfile_name, device=device)
        sides.training(
            data_loader_train,
            data_loader_val,
            num_epochs=opt.num_epochs,
            loss=opt.loss,
            burnin=opt.burnin
            )

    elif opt.mode == 'styleImages':
        from trainStyleImage import TrainerStyleImages
        style = TrainerStyleImages(opt, logfile_name, device=device)
        style.training(
            data_loader_train,
            data_loader_val,
            num_epochs=opt.num_epochs,
            loss=opt.loss,
            burnin=opt.burnin)

    elif opt.mode in ['styleLinear', 'styleMulti']:
        from trainStyleSignal import TrainerStyle
        style = TrainerStyle(opt, logfile_name, device=device)
        style.training(
            data_loader_train,
            data_loader_val,
            num_epochs=opt.num_epochs,
            loss=opt.loss,
            burnin=opt.burnin)

    elif opt.mode == 'styleFull':
        from trainFull import TrainerFull
        style = TrainerFull(opt, logfile_name, device=device)
        style.training(
            data_loader_train,
            data_loader_val,
            num_epochs=opt.num_epochs,
            loss=opt.loss, burnin=opt.burnin)

    else:
        raise NotImplementedError
        
        
