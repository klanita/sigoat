import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
import h5py

class UNetDataset(Dataset):
    """
        Data loader for style transfer and sided reconstruction.
    """
    def __init__(
            self, 
            file_name,
            split,
            input_modality='linear',
            output_modality='multi'
        ):
        """
        Args:
        file_name: str
            Full path to the synthetic dataset.
        input_modality: str (default: 'recon_linear')
            Name of the modality (of synthetic data) to use in the dataset.
        """        
        if 'syn' in file_name:
            scale = -1
        else:
            scale = 1
            
        h5_fh_in = h5py.File(f"{file_name}_{split}_{input_modality}_sigmat_multisegment.h5", 'r') 
        self.input = scale*h5_fh_in['BackProjection'][:]
        
        h5_fh_out = h5py.File(f"{file_name}_{split}_{output_modality}_sigmat_multisegment.h5", 'r') 
        self.output = scale*h5_fh_out['BackProjection'][:]
        
        print(self.input.shape, self.output.shape)
        
    def __getitem__(self, index):
        img_in = torch.Tensor(self.input[index])

        img_in /= torch.max(img_in)
        img_in = torch.clip(img_in, -0.2, 2)
        
        img_out = torch.Tensor(self.output[index])
        img_out /= torch.max(img_out)
        img_out = torch.clip(img_out, -0.2, 2)
        
        return img_in, img_out, index

    def __len__(self):
        return len(self.input)
    
    
class DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        opt
        ):
        """
        :param directory: directory with "TYPE" placeholder being replaced with "train", "val", "test"
        """
        super().__init__()
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.opt = opt
        self.batch_size = self.opt.batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_set = UNetDataset(
                self.opt.file_in,
                split='train',
                )

            self.val_set = UNetDataset(
                self.opt.file_in,
                split='val',
                )

            print(''.join(['-']*50))
            print(f"Number of train points: {len(self.train_set):,}")
            print(f"Number of validation points: {len(self.val_set):,}")
            print(''.join(['-']*50))
            
        if stage in (None, "test", "prediction"):
            self.test_set = UNetDataset(
                self.opt.file_in,
                split='test',
                )
            print(''.join(['-']*50))
            print(f"Number of test points: {len(self.test_set):,}")
            print(''.join(['-']*50))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.opt.num_workers,
            prefetch_factor=1,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.opt.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.opt.num_workers,
        )
