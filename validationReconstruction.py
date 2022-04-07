import h5py
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import time
import sys
sys.path.append("/home/anna/OptoAcoustics/")

from reconstructions import ReconstructionBenchmarks
from reconstructions import show_resonstruction
from ReconstructionBP import get_normal_signal
import os

import argparse
import json

idx_use = list(range(524, 1480))

def parse_args():
    parser = argparse.ArgumentParser(description='Validation Reconstruction')
    parser.add_argument('--folder', type=str)
    parser.add_argument('--tgtfolder', type=str, default='./MIDL/')
    parser.add_argument('--geometry', type=str, default='multi')
    parser.add_argument('--data', type=str, default='Real')
    parser.add_argument('--mode', type=str, default='signal_with_RC')    
    parser.add_argument('--nimgs', type=int, default=1)
    parser.add_argument('--scale_val', type=float, default=0)
    parser.add_argument('--subset', type=int, default=0)
    args = parser.parse_args()
    print(args)
    return args

def get_signal_vec(
    file,
    geometry='multi',
    mode='signal_with_denoise',
    scale=True,
    img_idx=0,
    scale_val=0.7,
    show=False
    ):
    # idx_signal = list(range(524, 1480))
    if geometry == 'linear':
        signal = file[mode][img_idx][:, 64:-64]
    else:        
        signal = file[mode][img_idx]
        if scale_val > 0:            
            signal[:, 64:-64] /= np.max(signal[:, 64:-64])
            signal[:, :64] = scale_val*signal[:, :64]/np.max(signal[:, :64])
            signal[:, -64:] = scale_val*signal[:, -64:]/np.max(signal[:, -64:])
            signal = np.clip(signal, -0.75, 1)

    signal /= np.max(signal)
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(signal, cmap='RdBu', interpolation='none')
        plt.colorbar()
    
    return signal


if __name__ == "__main__":
    opt = parse_args()
    file_in = f'{opt.folder}/{opt.data}.h5'

    file = h5py.File(file_in, 'r')
    file.keys()
    Rec = ReconstructionBenchmarks(geometry=opt.geometry)

    if opt.geometry == 'multi':
        scales = [0]
    else:
        scales = [0]

    for scale_val in scales:
        out_name = f'{opt.data}_{opt.geometry}_{opt.mode}'
        if (opt.geometry == 'multi') and (scale_val > 0):
            out_name = f'{out_name}_scaled'            
            
        tgt_folder = f'{opt.tgtfolder}/Reconstructions/{out_name}/'

        if not os.path.exists(tgt_folder):
            os.makedirs(tgt_folder)

        sigmat_size = [256, 256]    
        compression_lvl = 9 # pick between 0 and 9

        modes_list = ['BackProjection', 'ElasticNet 1e-5']
        fname_h5 = f'{tgt_folder}/{out_name}.h5'
        
        print('Creating file: %s' % fname_h5)
        data = {}
        with h5py.File(fname_h5, 'w', libver='latest') as h5_fh:
            for mode in modes_list:
                data[mode] =\
                    h5_fh.create_dataset(mode,
                    shape=[opt.nimgs] + sigmat_size, 
                    dtype=np.float32, chunks=tuple([1] + sigmat_size),
                    compression='gzip', compression_opts=compression_lvl)

        if opt.subset == 1:
            imdgs_idx = [1, 9, 12, 13, 22, 23, 26]
        else:
            imdgs_idx = range(opt.nimgs)

        with h5py.File(fname_h5, 'a', libver='latest') as h5_fh:
            for img_idx in imdgs_idx:
                print('Img_idx:', img_idx)
                start = time.time()
                signal = get_signal_vec(
                    file,
                    geometry=Rec.geometry,
                    mode=opt.mode,
                    scale_val=opt.scale_val,
                    img_idx=img_idx)

                reconstrutctions = {
            #         'Atb': Rec.reconstruction_Atb(signal),
                    'BackProjection': Rec.reconstruction_BP(
                        signal)[::-1],    
            #         'LSQR': RecLinear.reconstruction_lsqr(signal),
            #         'Lasso': RecLinear.reconstruction_linreg(signal, 'Lasso', alpha=1e-6),
                    # 'Ridge': RecLinear.reconstruction_linreg(signal, 'Ridge', alpha=1e-2),
                    'ElasticNet 1e-5': Rec.reconstruction_linreg(
                        get_normal_signal(signal[idx_use]), 'ElasticNet', alpha=1e-5),
                    # 'ElasticNet 0.25*1e-4': Rec.reconstruction_linreg(
                    #     signal, 'ElasticNet', alpha=0.25*1e-4),
            #         'TotalVariation': RecLinear.reconstruction_TV(signal),
                }      
                for mode in reconstrutctions.keys():
                    h5_fh[mode][img_idx] = reconstrutctions[mode]
                    
                show_resonstruction(
                    reconstrutctions,
                    fname=f'{tgt_folder}/Img_{img_idx}')
                print(f"Total time: {(time.time() - start)/60:.2f} min\n")