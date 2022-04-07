import sys
import torch
from ReconstructionBP import *
import numpy as np
import torchvision.utils as vutils
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd

mse_loss = torch.nn.MSELoss(reduction='mean')

def scale_img(x):
    x = x/np.max(x)
    x = x - np.min(x)
    return x/np.max(x)

def reconstructionBP(input_signal, left=None, right=None):    
    if len(input_signal.shape) == 3:
        input_signal = input_signal[0]

    sigMatVecInput = np.zeros([2030, 256])
    sigMatVecInput[idx_signal, 64:-64] = input_signal

    if not (left is None):
        sigMatVecInput[idx_signal, :64] = left
    if not (right is None):
        sigMatVecInput[idx_signal, -64:] = right

    recon_multi_input = scale_img(backProject(sigMatVecInput))
    # recon_multi_input = torch.tensor(recon_multi_input).clone().detach().unsqueeze(0).unsqueeze(0)
    # full_reconstruction = torch.tensor(sigMatVecInput).clone().detach().unsqueeze(0).unsqueeze(0)
    return recon_multi_input

def get_stat(signal):
    return [signal.shape, signal.min().item(),\
        signal.max().item(), signal.mean().item(), signal.std().item()]


def signal_to_device(signal_syn, signal_real, device):
    signal_syn = signal_syn.to(device)
    # .unsqueeze(1)
    signal_real = signal_real.to(device)
    # .unsqueeze(1)
    return signal_syn, signal_real

def get_accuracy(output, labels, device):
    predicted = torch.zeros(output.shape, device=device)
    predicted[output > 0.5] = 1
    total = labels.cpu().size(0)                
    correct = (predicted == labels).sum().item()
    return correct/total

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def FeatureMatching(x, y):
    mean_x = x.mean(0)
    std_x = x.std(0)
    mean_y = y.mean(0)
    std_y = y.std(0)
    return 0.5*(mse_loss(mean_x, mean_y) + mse_loss(std_x, std_y))

def MMDloss(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def split_output(x):
    return x[:, :, :, :64], x[:, :, :, 64:-64,], x[:, :, :, -64:]

def pretty_batch(img, max_imgs=8):
    if img.size(2) == img.size(3):
        x = torch.rot90(img, 1, [2,3])
    else:
        x = img
    return vutils.make_grid(x[:max_imgs], padding=10, normalize=True)

def show_resonstruction(
    imgs_list,
    titels=['Target', 'Reconstruction'],
    cmap='gray',
    fname=None,
    figsize=(7, 3)):

    if len(imgs_list) < 4:
        fig, ax = plt.subplots(1, len(imgs_list), figsize=figsize)
        for i in range(len(imgs_list)):
            ax[i].imshow(imgs_list[i], cmap=cmap)
            # , cmap='seismic'
            ax[i].set_title(titels[i], fontsize=5, fontweight='bold')
            ax[i].xaxis.set_tick_params(labelsize=3)
            ax[i].yaxis.set_tick_params(labelsize=3)
    else:        
        n1 = int(np.ceil(len(imgs_list)/2))
        n2 = 2
        figsize = (n2*2.25, n1*2.25)
        fig, ax = plt.subplots(n1, n2, figsize=figsize)
        l = 0
        for i in range(n1):
            for j in range(n2):
                if l < len(imgs_list):
                    ax[i, j].imshow(imgs_list[l], cmap=cmap)
                    # , cmap='seismic'
                    ax[i, j].set_title(titels[l], fontsize=5, fontweight='bold')
                    ax[i, j].xaxis.set_tick_params(labelsize=3)
                    ax[i, j].yaxis.set_tick_params(labelsize=3)
                else:
                    ax[i, j].axis('off')
                    ax[i, j].axis('equal')

                l += 1
    
    plt.tight_layout()
    if fname:
        plt.savefig(f'{fname}.png', format='png', dpi=300)
        plt.close(fig)
    else:
        plt.show()
