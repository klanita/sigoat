from options import parse_args

import torch
import numpy as np

from dataLoader import UnpairedDataset
from model import * 

from utils import *
from datetime import date
import h5py

import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Validation Signal')        
    parser.add_argument('--prefix', type=str,
        default='')
    parser.add_argument('--pretrained_sides', type=str,
        default='/home/anna/style_results/adv2021-06-21_chunk-1_sides/')
    parser.add_argument('--pretrained_style', type=str,
        default='/home/anna/style_results/deeplatent2021-06-09_chunk-1/')    
    parser.add_argument('--mode', type=str, default='sides_old_pipeline')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--epoch', type=str, default='')
    parser.add_argument('--tgt_dir', type=str, 
        # change it to yours
        default="/home/anna/OptoAcoustics/validation/",
        help="Target directory to save the model.")

    args = parser.parse_args()

    return args

class Validation:
    def __init__(
        self,
        mode='sidesTwo',
        device='cuda:0',
        prefix='',
        pretrained_style='/home/anna/style_results/deeplatent2021-06-09_chunk-1/',
        pretrained='/home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/',
        epoch=''
        ):

        self.device = device
        self.mode = mode        
        self.epoch = epoch
        
        self.load_models(pretrained_style, pretrained)

        style = pretrained_style.split('/')[-1]
        self.logfile_imgs = f'{opt.tgt_dir}/{prefix}{mode}{self.epoch}_{style}_{str(date.today())}/'
        if not os.path.exists(self.logfile_imgs):
            os.makedirs(self.logfile_imgs)
        print('Images will be saved to:\t', self.logfile_imgs, '\n')

    def load_models(self, pretrained_style, pretrained):        
        assert not (pretrained_style is None)        
        if self.mode == 'styleMulti':
            self.StyleNet = StyleNetwork('multi').to(self.device)
            self.StyleNet.load_state_dict(torch.load(f'{pretrained_style}/Style{self.epoch}.pth',
                map_location=torch.device(self.device)))
            self.StyleNet.eval()
            print(f"Loaded STYLE network from:\t{pretrained_style}/Style{self.epoch}.pth")
        elif self.mode == 'styleFull':
            normalization = pretrained_style.split('/')[-1].split('_')[-1]
            self.StyleNet = FullNetwork(
                'linear', normalization=normalization
            ).to(self.device)
            self.StyleNet.load_state_dict(
                torch.load(f'{pretrained_style}/Style{self.epoch}.pth',
                map_location=torch.device(self.device)))
            self.StyleNet.eval()
            print(f"Loaded FULL network from:\t{pretrained_style}/Style{self.epoch}.pth")
        else:
            # if 'Ablation' in pretrained_style:
            self.StyleNet = StyleNetworkAblation(
                'linear', normalization='instance').to(self.device)
            # else:
            #     self.StyleNet = StyleNetwork(-1).to(self.device)            
            self.StyleNet.load_state_dict(torch.load(f'{pretrained_style}/Style{self.epoch}.pth',
                map_location=torch.device(self.device)))
            self.StyleNet.eval()
            print(f"Loaded STYLE network from:\t{pretrained_style}/Style{self.epoch}.pth")

            if self.mode == 'sidesTwo':
                assert not (pretrained is None)
                self.SidesNet = [ReconstructionNetwork().to(self.device),
                    ReconstructionNetwork().to(self.device)]
                self.SidesNet[0].load_state_dict(
                    torch.load(f"{pretrained}/Reconstruction_left{self.epoch}.pth",
                    map_location=torch.device(self.device)))            
                self.SidesNet[0].eval()

                self.SidesNet[1].load_state_dict(
                    torch.load(f"{pretrained}/Reconstruction_right{self.epoch}.pth",
                    map_location=torch.device(self.device)))            
                self.SidesNet[1].eval()
                print(f"Loaded SIDES network from:\t{pretrained}/") 
            else:
                if self.mode == 'sides_old':            
                    old_model = '/home/anna/style_results/sim_training_l2only_sides_burnin500_2021-04-04__fader.pth.tar'
                    self.SidesNet = torch.load(old_model, map_location=torch.device(self.device))                   
                else:
                    self.SidesNet = FaderNetwork().to(self.device)
                self.SidesNet.load_state_dict(torch.load(pretrained + f'/sides{self.epoch}.pth',
                    map_location=torch.device(self.device)))
                self.SidesNet.eval()
                print(f"Loaded SIDES network from:\t{pretrained}/sides{self.epoch}.pth")     

    def signal_complete(self, raw_signal, use_style=True):
        with torch.no_grad():
            if self.mode == 'styleFull':
                signal_with_denoise = self.StyleNet(
                    raw_signal, real=False, full=True).detach().cpu()
                signal_with_RC = signal_with_denoise.clone()                
                signal_with_RC[:, :, :, 64:-64] = raw_signal
            else:
                if use_style:
                    signal_denoise = self.StyleNet(raw_signal, real=False)
                else:
                    signal_denoise = raw_signal
                if self.mode == 'styleMulti':
                    signal_with_denoise = signal_denoise
                    signal_with_RC = signal_with_denoise.clone()                
                elif self.mode == 'sidesTwo':
                    reconsts_left = self.SidesNet[0](signal_denoise)
                    reconsts_right = self.SidesNet[1](signal_denoise)
                    signal_with_RC = torch.cat([reconsts_left, reconsts_right], dim=3).detach()
                    signal_with_RC[:, :, :, 64:-64] = raw_signal
                    signal_with_denoise = signal_with_RC.clone()                
                    signal_with_denoise[:, :, :, 64:-64] = signal_denoise
                else:
                    left_rec, _, right_rec = split_output(self.SidesNet(signal_denoise))
                    signal_with_RC = torch.cat(
                        (left_rec, raw_signal, right_rec), 3).to(self.device)
                    signal_with_denoise = torch.cat(
                        (left_rec, signal_denoise, right_rec), 3).to(self.device)

        return signal_with_RC, signal_with_denoise

    
    def save_h5(self, signal_tgt, fname):
        fname_h5 = f'{self.logfile_imgs}/{fname}.h5'        

        num_images = signal_tgt.shape[0]
        sigmat_size = [956, 256]    
        compression_lvl = 9 # pick between 0 and 9
        
        print('Creating file: %s' % fname_h5)

        if self.mode == 'styleMulti':
            signal_with_RC, signal_with_denoise =\
                self.signal_complete(signal_tgt.to(self.device))
        else:
            signal_with_RC, signal_with_denoise =\
                self.signal_complete(signal_tgt[:, :, :, 64:-64].to(self.device))
            
        modes_list = ['sigmat_multisegment', 'signal_with_RC', 'signal_with_denoise']
        data = {}
        with h5py.File(fname_h5, 'w', libver='latest') as h5_fh:
            for mode in modes_list:
                data[mode] =\
                    h5_fh.create_dataset(mode,
                    shape=[num_images] + sigmat_size, 
                    dtype=np.float32, chunks=tuple([1] + sigmat_size),
                    compression='gzip', compression_opts=compression_lvl)
                    
        for i in range(num_images):
            with h5py.File(fname_h5, 'a', libver='latest') as h5_fh:
                h5_fh['sigmat_multisegment'][i] = signal_tgt[i].numpy()
                h5_fh['signal_with_RC'][i] = signal_with_RC[i][0].cpu().numpy()
                h5_fh['signal_with_denoise'][i] = signal_with_denoise[i][0].cpu().numpy()
    
        for i_im in [0, 12]:
            show_signal(
                fname_h5,
                im=i_im,
                fname=f'{self.logfile_imgs}/{fname}_Signal_{i_im}.png'
            )


def norm(img, scale=1):
    if scale == 1:
        img[:, 64:-64] /= np.max(img[:, 64:-64])
        img[:, :64] = 0.7*img[:, :64]/np.max(img[:, :64])
        img[:, -64:] = 0.7*img[:, -64:]/np.max(img[:, -64:])
        img = np.clip(img, -0.75, 1)
#     img = (img - np.mean(img))/np.std(img)
    
    img /= np.max(img)
    
    img = img[150:600]
    n1 = img.shape[0]
    if img.shape[1] == 128:
        img = np.concatenate(
            [np.ones([n1, 64]), img, np.ones([n1, 64])], axis=1)
    print(img.min(), img.max())
    return img

def show_signal(file_name, im=0, fs=9, fname=None):
    file = h5py.File(file_name, 'r')
    n = len(file.keys())+2
    fig, ax = plt.subplots(1, n, figsize=(n*4.25-3.15, 2*4.25), sharex=True, sharey=True)
    print(file.keys())
           
    ax[0].imshow(norm(file['sigmat_multisegment'][im], scale=0), cmap='gray', vmin=-1, vmax=1)
    ax[0].set_title('Multi GT', fontsize=fs, fontweight='bold')
    
    ax[1].imshow(norm(file['sigmat_multisegment'][im][:, 64:-64], scale=0), cmap='gray', vmin=-1, vmax=1)
    ax[1].set_title('Linear GT', fontsize=fs, fontweight='bold')
    
    
    ax[2].imshow(norm(file['signal_with_denoise'][im][:, 64:-64], scale=0), cmap='gray', vmin=-1, vmax=1)
    ax[2].set_title('Linear denoise', fontsize=fs, fontweight='bold')
    
    ax[3].imshow(norm(file['signal_with_denoise'][im], scale=0), cmap='gray', vmin=-1, vmax=1)
    ax[3].set_title('Linear impute', fontsize=fs, fontweight='bold')
    
    ax[4].imshow(norm(file['signal_with_RC'][im]), cmap='gray', vmin=-1, vmax=1)
    ax[4].set_title('Linear impute with RC', fontsize=fs, fontweight='bold')
     
    
    for i in range(n):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_axis_off()
    
#     divider = make_axes_locatable(ax[5])
#     cax = divider.append_axes('right', size='10%', pad=0.05)
#     fig.colorbar(im1, cax=cax, orientation='vertical')
    
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)
    
    if fname:
        plt.savefig(fname, format='png', dpi=300)
        plt.show()


if __name__ == "__main__":
    opt = parse_args()

    model = Validation(
        prefix=opt.prefix,
        mode=opt.mode,
        device=opt.device,
        pretrained=opt.pretrained_sides,
        pretrained_style=opt.pretrained_style,
        epoch=opt.epoch
    )

    with open(f'{model.logfile_imgs}/commandline_args.txt', 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    file_in = '/home/anna/dlbirhoui_data/parsed_simulated_ellipsesSkinMask_mgt_ms_ring_256_ratio_09_1_20210412.h5'
    target = '/home/anna/dlbirhoui_data/arm.h5'

    # file_in = ''
    # target = ''

    dataset = UnpairedDataset(
        file_in,
        file_name_real=target,
        geometry='multisegment',
        validation=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1024,
        shuffle=False,
    )

    dataiter = iter(data_loader)

    syn_tgt, real_tgt, img_num = dataiter.next()

    model.save_h5(real_tgt, 'Real')
    # model.save_h5(syn_tgt, 'Syn')