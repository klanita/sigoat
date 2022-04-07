import h5py
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

import skimage
import cv2

import torchvision.utils as vutils
import torch

path = './MIDL/validation/'

methods = ['BackProjection' , 'ElasticNet 1e-5']

cmap='gray'

files_list = {
    'Multi': f'/home/anna/dlbirhoui_data//Reconstructions/arm_multi_sigmat_multisegment//arm_multi_sigmat_multisegment.h5',
    'Linear': f'/home/anna/dlbirhoui_data//Reconstructions/arm_linear_sigmat_multisegment//arm_linear_sigmat_multisegment.h5',

    'Side impute': f'{path}/NewLinearInput_sides_old_pipeline__2022-02-09/Reconstructions/Real_multi_signal_with_denoise/Real_multi_signal_with_denoise.h5',
    'Side impute with RC': f'{path}/NewLinearInput_sides_old_pipeline__2022-02-09/Reconstructions/Real_multi_signal_with_RC/Real_multi_signal_with_RC.h5',
        
    'Unet': f'./MIDL/from_firat_1024//unet_ellipsesSkinMask_20210412_lin2Ms_scaleclip_MAE.h5',
    
    # benchmarks
    'Side impute with RC wo Style': f'{path}/Behnch1024_woStyle_L2_full_styleNone_2022-02-10/Reconstructions/Real_multi_signal_with_RC/Real_multi_signal_with_RC.h5',
    # 'Side impute with RC wo Style Syn': f'./MIDL/Behnch1024_woStyleSyn_L2_full_styleNone_2022-02-10//Reconstructions/Real_multi_signal_with_RC/Real_multi_signal_with_RC.h5',
    'Side impute with RC Two Sides': f'{path}/NewSides_sidesTwo__2022-02-10/Reconstructions/Real_multi_signal_with_RC/Real_multi_signal_with_RC.h5'
}


def scale(im):
    sorted_int_values = np.sort(np.reshape(im, [-1]))
    low5, high95 = int(len(sorted_int_values) * 0.025), int(len(sorted_int_values) * 0.975)
#     im_clipped = np.clip(im, a_min=sorted_int_values[low5], a_max=sorted_int_values[high95])
    im_clipped = im
    im_clipped /= np.max(im_clipped)
    im_clipped = np.clip(im_clipped, -0.2, 2)
    return im_clipped


def figure_plot(imgs_dict, fname=None, tgt_metric='spearman',
    scale_bar=True, colorbar=False, figsize=(5, 10)):
    
    modalities = list(imgs_dict.keys())
    n2 = len(modalities)
    methods = list(imgs_dict[modalities[0]].keys()) # reconstruction methods, e.g. EN
    n1 = len(methods)
    
    metrics_dict = {}
#     for modality in modalities:
#         ssim_dict[modality] = {}
         
    figsize = (n2*2.25-3, n1*2.25)
    fig, ax = plt.subplots(n1, n2, figsize=figsize, sharey=False, sharex=False)
    l = 0
    scores_names = ['SSIM', 'MSE', 'MAE', 'SNR', 'pearson', 'R2', 'PSNR']
    imgs_correct = {}
    for i, method in enumerate(methods):
        gt = np.rot90(imgs_dict['Multi'][method], 1, [0,1])
#         gt = (gt - np.mean(gt))/np.std(gt)        
        gt = scale(gt)        
        metrics_dict[method] = {}
        imgs_correct[method] = {}
        for score in scores_names:
            metrics_dict[method][score] = {}
        for j, modality in enumerate(modalities):            
#             metrics_dict[method][modality] = {}            
            pred = imgs_dict[modality][method].copy()
            if modality != 'Unet':
                pred = np.rot90(pred, 1, [0,1])
            # else:
            pred = pred[:, ::-1]
                
            pred = scale(pred)
            imgs_correct[method][modality] = pred
            
            ssim_const = ssim(pred, gt,
                  data_range=gt.max() - gt.min())
            mse = mean_squared_error(gt, pred)
            mae = np.mean(abs(gt - pred))
#             s = spearmanr(gt.reshape(-1), pred.reshape(-1))[0]
            s = skimage.metrics.peak_signal_noise_ratio(gt, pred, data_range=None)
            psnr = cv2.PSNR(gt, pred)
        
            p = pearsonr(gt.reshape(-1), pred.reshape(-1))[0]
            r2 = r2_score(gt.reshape(-1), pred.reshape(-1))
            
            metrics_dict[method]['SSIM'][modality] = ssim_const
            metrics_dict[method]['MSE'][modality] = mse
            metrics_dict[method]['MAE'][modality] = mae
            metrics_dict[method]['SNR'][modality] = s
            metrics_dict[method]['pearson'][modality] = p
            metrics_dict[method]['R2'][modality] = r2
            metrics_dict[method]['PSNR'][modality] = psnr
            
            
#             mse_const[method][modality] = mse
#             ax[i, j].set_title(f"{modality}", fontsize=9, fontweight='bold')
            
            if scale_bar:
                if (i == 0) and (j == len(modalities)-1):
                    pred[220:221][:, 180:180+50] = -1
                    pred[218:224][:, 180] = -1
                    pred[218:224][:, 180+50] = -1
            
            im1 = ax[i, j].imshow(pred, cmap=cmap, vmin=-1, vmax=1)
    
            if i == 0:
                ax[i, j].set_title(
                    f'{modality}'+ f'\n{tgt_metric}={metrics_dict[method][tgt_metric][modality]:.2f}', fontsize=9)
            else:
                ax[i, j].set_title(
                    f'{tgt_metric}={metrics_dict[method][tgt_metric][modality]:.2f}', fontsize=9)

            if j == 0:
                ax[i, j].set_ylabel(method, fontsize=9, fontweight='bold')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            
    
    if colorbar:
        divider = make_axes_locatable(ax[i, j])
        cax = divider.append_axes('right', size='10%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.005)
#     cax = plt.axes([0.85, 0.1, 0.075, 0.8])
#     plt.colorbar()
#     plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=None)
    if fname:
        plt.savefig(fname, format='png', dpi=300)
#         plt.close()
    return imgs_correct, metrics_dict


def read_batch(imgs_dict, cmap='gray'):  
    batch_dict = {'BackProjection': [], 'ElasticNet 1e-5': []}
    for modality in imgs_dict.keys():
        for rec_meth in imgs_dict[modality].keys():
            pred = imgs_dict[modality][rec_meth]
            if names[modality] != 'Unet':
                pred = np.rot90(pred, 1, [0,1])
            else:
                pred = pred[:, ::-1]
            pred = scale(pred)    
#             print(names[modality], rec_meth)
            batch_dict[rec_meth].append(pred)
    
    batch_img = {}
    for rec_meth in batch_dict.keys():
        img_batch = torch.Tensor(batch_dict[rec_meth]).unsqueeze(1)        
        img = vutils.make_grid(img_batch, 
                               padding=5, normalize=True, nrow=len(files)).numpy()[0]
        batch_img[rec_meth] = img
#         plt.figure(figsize=(20, 20))
#         plt.imshow(img, cmap=cmap)
#         plt.title(rec_meth)
    return batch_img


def print_to_latex(df):
    for method in ['Linear GT', 'Side impute', 'Side impute with RC', 'Unet', 'Side impute with RC wo Style', 'Side impute with RC wo Style Syn', 'Side impute with RC Two Sides']:
        ssim = df['SSIM'].loc[method].mean()
        mse = df['MSE'].loc[method].mean()
        snr = df['SNR'].loc[method].mean()
        psnr = df['PSNR'].loc[method].mean()

        ssim_std = df['SSIM'].loc[method].std()
        mse_std = df['MSE'].loc[method].std()
        snr_std = df['SNR'].loc[method].std()
        psnr_std = df['PSNR'].loc[method].std()

        print('\\textbf{'+method+'} &'\
            + f"{ssim:.2f} $\pm$ {ssim_std:.2f} & {mse:.4f} $\pm$ {mse_std:.4f} &  {snr:.2f} $\pm$ {snr_std:.2f} &  {psnr:.2f} $\pm$ {psnr_std:.2f}"\
            +"\\\\")
        print('\cline{1-5}')

def get_imgs(reconstrutctions, im=0, cmap='gray'):
    rec_names = list(reconstrutctions.keys()) # ['BackProjection', 'ElasticNet 1e-5']
    imgs_dict = {}
    for mode in rec_names:
        imgs_dict[mode] = reconstrutctions[mode][im]
    return imgs_dict


if __name__ == "__main__":
    df_BP = pd.DataFrame() # BackProjection
    df_EN = pd.DataFrame() # ElasticNet

    batch_dict = {'BackProjection': [], 'ElasticNet 1e-5': []}

    # for i_im in selected_imgs:
    for i_im in range(100):
        imgs_dict = {}
        for benchmark in files_list:
            fname = files_list[benchmark]
            print(benchmark)
            imgs_dict[benchmark] = get_imgs(h5py.File(fname, 'r'), im=i_im)

        imgs_correct, metrics_dict = figure_plot(
            imgs_dict,
            tgt_metric='SSIM',
            fname=f'./PaperFigures1024/1024_all/Img_{i_im}_withunet.png'
        )

    quit()
        # batch_img = read_batch(imgs_dict)
        # for meth in batch_dict.keys():
        #     batch_dict[meth].append(batch_img[meth])

        # df_img = pd.DataFrame(metrics_dict['BackProjection'])
        # df_img['sample'] = i_im
        # df_BP = pd.concat([df_BP, df_img])
        
        # df_img = pd.DataFrame(metrics_dict['ElasticNet 1e-5'])
        # df_img['sample'] = i_im
        # df_EN = pd.concat([df_EN, df_img])

    df_EN['method'] = df_EN.index
    df_EN_mean = df_EN.groupby('method').mean().sort_values('PSNR', ascending=False).style.format('{:.4f}')

    df_BP['method'] = df_BP.index
    df_BP_mean = df_BP.groupby('method').mean().sort_values('PSNR', ascending=False).style.format('{:.4f}')
    
    print('====EN====')
    print(df_EN_mean)
    print_to_latex(df_EN)

    print('====BP====')
    print(df_BP_mean)
    print_to_latex(df_BP)

    cmap = 'gray'

    # for i, meth in enumerate(batch_dict.keys()):
    #     fig = plt.figure(figsize=(8, 11))
    #     img = vutils.make_grid(torch.Tensor(batch_dict[meth][:16]).unsqueeze(1), 
    #                            padding=5, normalize=True, nrow=1).numpy()[0]
        
    #     plt.imshow(img, cmap=cmap)
    #     ax = plt.gca()
    # #     ax[i].set_title(meth, fontsize=9, fontweight='bold')
    #     ax.set_xticks([])
    #     ax.set_yticks([])
        
    #     fname = f'../PaperFigures/Results_{meth}.pdf'
    #     plt.tight_layout()
    #     plt.savefig(fname, format='pdf')


    fig, ax = plt.subplots(1, 2, figsize=(16, 16), sharex=True, sharey=True)
    for i, meth in enumerate(batch_dict.keys()):
        img = vutils.make_grid(torch.Tensor(batch_dict[meth][:16]).unsqueeze(1), 
                            padding=5, normalize=False, nrow=1).numpy()[0]
        
        print(img.min(), img.max())
        im1 = ax[i].imshow(img, cmap=cmap, vmin=-0.2, vmax=1)
        ax[i].set_title(f'{meth}\n\nMulti GT  Linear GT ImpSide  ImpSide-RC  Unet  BM1 BM2 BM3 ', fontsize=9)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    
    # divider = make_axes_locatable(ax[i])
    # cax = divider.append_axes('right', size='10%', pad=0.05)
    # cb = fig.colorbar(im1, cax=cax, orientation='vertical', pad=0.01)
    # cb.set_ticks([])
    # # cb.ax.text(136,66,s='min', ha='center', va='top', fontsize=12, rotation='vertical', color='w')
    # # cb.ax.text(136,185,s='max', ha='center', va='bottom', fontsize=12, rotation='vertical', color='k')
        
    fname = f'./PaperFigures/SupFig1.pdf'
    # plt.tight_layout()
    plt.savefig(fname, format='pdf', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(8, 11), sharex=True, sharey=True)
    for i, meth in enumerate(batch_dict.keys()):
        img = vutils.make_grid(torch.Tensor(batch_dict[meth][16:32]).unsqueeze(1), 
                            padding=5, normalize=False, nrow=1).numpy()[0]
        
        print(img.min(), img.max())
        im1 = ax[i].imshow(img, cmap=cmap, vmin=-0.2, vmax=1)
        ax[i].set_title(f'{meth}\n\nMulti GT  Linear GT ImpSide  ImpSide-RC  Unet  ', fontsize=9)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    
    # divider = make_axes_locatable(ax[i])
    # cax = divider.append_axes('right', size='10%', pad=0.05)
    # cb = fig.colorbar(im1, cax=cax, orientation='vertical', pad=0.01)
    # cb.set_ticks([])
    # # cb.ax.text(136,66,s='min', ha='center', va='top', fontsize=12, rotation='vertical', color='w')
    # # cb.ax.text(136,185,s='max', ha='center', va='bottom', fontsize=12, rotation='vertical', color='k')
        
    fname = f'./PaperFigures/SupFig2.pdf'
    plt.tight_layout()
    plt.savefig(fname, format='pdf', dpi=300)