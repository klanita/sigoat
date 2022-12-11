import os
import math
import numpy as np
from scipy.sparse import random
from scipy.stats import rv_continuous
from functools import reduce
from operator import __add__
from torch.utils.data import Dataset
from scipy.sparse.linalg import lsqr
import torch
import h5py
from torchvision.utils import make_grid
import scipy.io as sio
import random
import time

from ReconstructionBP import *
# from trainVN import normalizeInput
from utils import scale_img

def calculate_split(idx_split = -1):
    max_split = 7 # maximum amount of splits available

    if (idx_split < -2) or (idx_split > max_split):
        print('ERROR! This chunk size does not exist!')
        raise NotImplementedError
    
    if idx_split == -1:
        return list(range(524, 1480))
    else:
        idx_start = 524 + idx_split*128
        if max_split != idx_split:
            idx_end = 524 + (idx_split+1)*128
            return list(range(idx_start, idx_end))
        else:
            idx_start = 1480 - 128
            return list(range(idx_start, 1480))

def FilterSignalTorch(sigMat,
                main_dir='/home/anna/dlbirhoui/',
                array_geometry='multisegmentCup',
                resolutionXY=256, 
                reconDimsXY=0.0256,
                speedOfSound=1525,
                fSampling=40e6,
                delayInSamples=61,
                nSamples=2028,
                lowCutOff=0.1e6,
                highCutOff=6e6,
                fOrder=3):
                
    array_dir = os.path.join(main_dir, 'util', 'arrayInfo', array_geometry+'.mat')
    arrayFile = sio.loadmat(array_dir)     

    transducerPos = arrayFile['transducerPos']    
    xSensor = np.transpose(transducerPos[:,0])
    ySensor = np.transpose(transducerPos[:,1])
    timePoints = np.arange(0, (nSamples)/fSampling, 1/fSampling) + delayInSamples/fSampling

    if sigMat.ndim != 3:
        sigMat = np.expand_dims(sigMat, axis=2)

    # filter sigMat (we can skip for now, doesnt change output a lot)
    sigMatF         = (-1)*sigMatFilter(sigMat, lowCutOff, highCutOff, fSampling, fOrder, 0.5)

#     normalize sigMat around 0\
    sigMatN         = torch.Tensor(sigMatNormalize(sigMatF))
    sigMatN /= sigMatN.abs().max()
    return sigMatN

def sigMatNormalizeTensor(signal, idx=range(64, 128+64)):
    d = len(signal.shape)-2
    signal = signal - signal.mean(dim=d).expand_as(signal)
    signal = signal/signal.std(dim=d).expand_as(signal)
    signal /= (signal[:, idx].abs()).max()

    return signal

def signalToPlot(signal, padding=30, pad_value=1):
    return make_grid(signal, 
        padding=padding, pad_value=pad_value, normalize=True, scale_each=True)

def signalToVec(signal):
    return torch.transpose(torch.Tensor(signal), 1, 0).reshape(-1, 1)

def signalToMat(signal, im_dim, im_signal):
    return signal.reshape(-1, im_dim, im_signal).transpose(2, 1)


def get_idx_signal(signal_shape, idx_use):
    mask = np.transpose(np.zeros(signal_shape)).reshape(-1, 1)
    mask[idx_use] = 1
    mask_im = np.transpose(mask.reshape(signal_shape[1], signal_shape[0]))
    idx_signal = np.where(np.sum(mask_im, axis=1) != 0)[0]
    mask_im[idx_signal, :] = 1
    mask = np.transpose(mask_im).reshape(-1, 1)
    idx_use = np.where(mask != 0)[0]
    return idx_signal, idx_use

class UnpairedDataset(Dataset):
    """
        Data loader for style transfer and sided reconstruction.
    """
    def __init__(
            self, 
            file_name_syn,
            file_name_real='/home/anna/dlbirhoui_data/arm.h5',
            input_modality_syn='sigmat_multisegment',
            input_modality_real='sigmat_multisegment',
            geometry=None,
            clip_norm=False,
            validation=False,
            dataset='real'
            # idx_signal = list(range(524, 1480))
        ):
        """
        Args:
        file_name_syn: str
            Full path to the synthetic dataset.
        file_name_real: str (default: '/home/anna/dlbirhoui_data/arm.h5')
            Full path to the real dataset.
        input_modality_syn: str (default: 'sigmat_multisegment')
            Name of the modality (of synthetic data) to use in the dataset.
        input_modality_real: str (default: 'sigmat_multisegment')
            Name of the modality to (of synthetic data) use in the dataset.
        # idx_signal: list (default: list(range(524, 1480))
        #     Range of signal to use.
        idx_split: list (defalt: -1)
            Chunk of the signal to take for the style transfer. -1 corresponds 
            to the whole signal. Maximum 14.
        """        
        idx_signal = calculate_split(-1)

        if 'ring' in geometry:
            self.input_modality_syn = 'sigmat_ring'
            self.input_modality_real = 'rawSignalsFull'
        else:
            self.input_modality_syn = input_modality_syn
            self.input_modality_real = input_modality_real

        self.file_name_syn = file_name_syn
        self.file_name_real = file_name_real
        
        # with h5py.File(self.file_name_syn, 'r', libver='latest') as h5_fh:
        h5_fh = h5py.File(self.file_name_syn, 'r') 
        self.signal_syn = h5_fh[self.input_modality_syn][:]
        self.len_syn = len(self.signal_syn)

        # with h5py.File(self.file_name_real, 'r', libver='latest') as h5_fh:
        h5_fh = h5py.File(self.file_name_real, 'r')
        self.signal_real = h5_fh[self.input_modality_real][:]
        self.len_real = len(self.signal_real)

        self.idx_signal = idx_signal
        self.clip_norm = clip_norm

        if self.file_name_real == './data/benchmark_invivo.h5':
            self.img_num = h5_fh['img_num'][:]
        else:
            self.img_num = None
        self.validation = validation

    def __getitem__(self, index):
        if self.validation:
            index_real = index % self.len_real
        else:
            index_real = random.randint(0, self.len_real - 1) % self.len_real
        # select only linear part        
        if 'ring' in self.input_modality_syn:
            idx_syn = range(256, 512)
            idx_real = range(0, 256)            
            idx_syn_linear = range(256+64, 512-64)
            idx_real_linear = range(0+64, 256-64)
            scale_real = 1
        else:
            idx_syn = range(0, 256)
            idx_real = range(0, 256)
            idx_syn_linear = range(0+64, 256-64)
            idx_real_linear = range(0+64, 256-64)
            scale_real = -1

        y_syn = self.signal_syn[index % self.len_syn][:, idx_syn][self.idx_signal]
        y_real = scale_real*self.signal_real[index_real][:, idx_real][self.idx_signal]

        y_syn = sigMatFilter(np.expand_dims(y_syn, axis=2))[:, :, 0]
        y_real = sigMatFilter(np.expand_dims(y_real, axis=2))[:, :, 0]
               
        y_syn = sigMatNormalizeTensor(torch.Tensor(y_syn)).unsqueeze(2)            
        y_real = sigMatNormalizeTensor(torch.Tensor(y_real)).unsqueeze(2)

        if self.img_num is None:
            idx_sample = index
        else:
            idx_sample = self.img_num[index][0]
        
        return y_syn.squeeze(2).unsqueeze(0), y_real.squeeze(2).unsqueeze(0), idx_sample

    def __len__(self):
        return max(len(self.signal_syn), len(self.signal_real))


class TestDataset(Dataset):
    """
        Data loader for style transfer and sided reconstruction.
    """
    def __init__(
            self, 
            file_name,
            input_modality='sigmat_multisegment',
            clip_norm=False,
        ):
        """
        Args:
        file_name: str
            Full path to the synthetic dataset.
        input_modality: str (default: 'sigmat_multisegment')
            Name of the modality (of synthetic data) to use in the dataset.
        """        
        idx_signal = calculate_split(-1)

        self.input_modality = input_modality
        self.file_name = file_name
        
        h5_fh = h5py.File(self.file_name, 'r') 
        self.signal = h5_fh[self.input_modality][:]
        self.len = len(self.signal)

        self.idx_signal = idx_signal
        self.clip_norm = clip_norm

        if self.file_name == './data/benchmark_invivo.h5':
            self.img_num = h5_fh['img_num'][:]
        else:
            self.img_num = None

        self.scale = -1
        if 'syn' in file_name:
            self.scale = 1

    def __getitem__(self, index):
        idx = range(0, 256)
        cutout_val = abs(self.signal[index][:524]).sum()
        signal_scaled = self.scale*self.signal[index][:, idx][self.idx_signal]
        signal_scaled = sigMatFilter(np.expand_dims(signal_scaled, axis=2))[:, :, 0]               
        signal_scaled = sigMatNormalizeTensor(torch.Tensor(signal_scaled)).unsqueeze(2)            

        if self.img_num is None:
            idx_sample = index
        else:
            idx_sample = self.img_num[index][0]
        
        return signal_scaled.squeeze(2).unsqueeze(0), cutout_val

    def __len__(self):
        return len(self.signal)
    
    
class UnpairedDatasetImages(Dataset):
    """
        Data loader for style transfer and sided reconstruction.
    """
    def __init__(
            self, 
            file_name_syn,
            file_name_real='/home/anna/dlbirhoui_data/arm.h5',
            input_modality_syn='recon_multisegment',
            input_modality_real='recon_linear',
            # input_modality_real='recon_multisegment',
        ):
        """
        Args:
        file_name_syn: str
            Full path to the synthetic dataset.
        file_name_real: str (default: '/home/anna/dlbirhoui_data/arm.h5')
            Full path to the real dataset.
        input_modality_syn: str (default: 'sigmat_multisegment')
            Name of the modality (of synthetic data) to use in the dataset.
        input_modality_real: str (default: 'sigmat_multisegment')
            Name of the modality to (of synthetic data) use in the dataset.
        # idx_signal: list (default: list(range(524, 1480))
        #     Range of signal to use.
        idx_split: list (defalt: -1)
            Chunk of the signal to take for the style transfer. -1 corresponds 
            to the whole signal. Maximum 14.
        """        

        self.file_name_syn = file_name_syn
        self.file_name_real = file_name_real
        self.input_modality_syn = input_modality_syn
        self.input_modality_real = input_modality_real
        
        h5_fh = h5py.File(self.file_name_syn, 'r') 
        self.img_syn = h5_fh[self.input_modality_syn][:]
        self.len_syn = len(self.img_syn)

        h5_fh = h5py.File(self.file_name_real, 'r')
        self.img_real = h5_fh[self.input_modality_real][:]
        self.len_real = len(self.img_real)
        
    def __getitem__(self, index):
        img_syn = np.flip(np.rot90(self.img_syn[index % self.len_syn], 1, (1,0)), axis=1)
        img_syn = torch.Tensor(img_syn)
        img_syn /= img_syn.abs().max()

        index_real = random.randint(0, self.len_real - 1) % self.len_real
        # img_real = torch.Tensor(np.copy(np.rot90(self.img_real[index_real], k=2)))
        img_real = torch.Tensor(self.img_real[index_real])
        img_real /= img_real.abs().max()

        return img_syn.unsqueeze(0), img_real.unsqueeze(0)

    def __len__(self):
        return max(len(self.img_syn), len(self.img_real))
