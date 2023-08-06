from torch.utils.data import Dataset
import h5py
from ReconstructionBP import *
import torch

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

def sigMatNormalizeTensor(signal, idx=range(64, 128+64)):
    d = len(signal.shape)-2
    signal = signal - signal.mean(dim=d).expand_as(signal)
    signal = signal/signal.std(dim=d).expand_as(signal)
    signal /= (signal[:, idx].abs()).max()

    return signal

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

        self.input_modality_syn = input_modality_syn
        self.input_modality_real = input_modality_real

        self.file_name_syn = file_name_syn
        
        # with h5py.File(self.file_name_syn, 'r', libver='latest') as h5_fh:
        h5_fh = h5py.File(self.file_name_syn, 'r') 
        self.signal_syn = h5_fh[self.input_modality_syn]#[:]
        self.len_syn = len(self.signal_syn)

        self.idx_signal = idx_signal
        self.clip_norm = clip_norm

        self.validation = validation

    def __getitem__(self, index):
        y_syn = self.signal_syn[index % self.len_syn][self.idx_signal]
        y_syn = sigMatFilter(np.expand_dims(y_syn, axis=2))[:, :, 0]               
        y_syn = sigMatNormalizeTensor(torch.Tensor(y_syn)).unsqueeze(2)            

        return y_syn.squeeze(2).unsqueeze(0)

    def __len__(self):
        return len(self.signal_syn)