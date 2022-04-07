import sys
# sys.path.append("/home/anna/dlbirhoui/")
# sys.path.append("/home/anna/dlbirhoui/fadern/")

import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from utils import *
from dataLoader import *
from model import *

real_label = 1.
syn_label = 0.
criterion = nn.BCELoss()

def classifier_step(netD, with_style, signal_truth, weight_grad_adv, device='cuda:0', sigma=0.05):
    noise = torch.normal(0, sigma, with_style.shape, device=device)
    outputD_gen = netD(with_style + noise).view(-1)
    b_size = outputD_gen.size(0)

    label_truth = torch.full((b_size, ), real_label, dtype=torch.float, device=device)
    label_gen = torch.full((b_size, ), syn_label, dtype=torch.float, device=device)

    errD_gen = criterion(outputD_gen, label_gen)
    adversary_penalty = torch.autograd.grad(
        errD_gen,
        outputD_gen,
        create_graph=True)[0].pow(2).mean()

    noise = torch.normal(0, sigma, signal_truth.shape, device=device)
    outputD = netD(noise + signal_truth).view(-1)
    errD = criterion(outputD, label_truth)
                    
    adversary_penalty += torch.autograd.grad(
        errD,
        outputD,
        create_graph=True)[0].pow(2).mean()

    acc = get_accuracy(torch.cat((outputD, outputD_gen), dim=0),\
                        torch.cat((label_truth, label_gen), dim=0), device)

    err = errD_gen + errD + weight_grad_adv * adversary_penalty    

    return err, acc, adversary_penalty.item()


class TrainerFull:
    def __init__(
        self,
        opt,
        logfile_name,
        device=None
        ):

        self.logfile_name = logfile_name
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.StyleNet = FullNetwork(
            'linear', normalization=opt.normalization
            ).to(self.device)

        latent_dim = self.StyleNet.latent_dim

        self.netD_latent = DiscriminatorLatent(latent_dim).to(self.device)
        self.netD_denoise = Discriminator(patch=opt.patch).to(self.device)
        self.netD_noise = Discriminator(patch=opt.patch).to(self.device)

        if not (opt.pretrained) is None:
            print(f'Loading pretrained model from: {opt.pretrained}')
            self.StyleNet.load_state_dict(torch.load(f'{opt.pretrained}/Style.pth'))            
            self.netD_latent.load_state_dict(torch.load(f'{opt.pretrained}/netD_latent.pth'))
            self.netD_denoise.load_state_dict(torch.load(f'{opt.pretrained}/netD_denoise.pth'))
            self.netD_noise.load_state_dict(torch.load(f'{opt.pretrained}/netD_noise.pth'))
            
            self.StyleNet.train()
            self.netD_latent.train()
            self.netD_denoise.train()
            self.netD_noise.train()
       
            # Setup Adam optimizers for both G and D
        self.optimizerD_noise = optim.Adam(
            self.netD_noise.parameters(),
            lr=opt.lr*opt.adv_lr_mult, 
            betas=(opt.beta1, 0.999),
            weight_decay=opt.weight_decay)

        self.optimizerD_denoise = optim.Adam(
            self.netD_denoise.parameters(),
            lr=opt.lr*opt.adv_lr_mult, 
            betas=(opt.beta1, 0.999),
            weight_decay=opt.weight_decay)

        self.optimizerD_latent = optim.Adam(
            self.netD_latent.parameters(),
            lr=opt.lr*1e-2,
            betas=(opt.beta1, 0.999),
            weight_decay=opt.weight_decay)

        self.optimizerS = optim.Adam(
                self.StyleNet.parameters(), 
                lr=opt.lr, 
                betas=(opt.beta1, 0.999))

        self.scheduler_G = lr_scheduler.StepLR(
            self.optimizerS, 
            step_size=opt.lr_decay_iters, 
            gamma=0.1)
        self.scheduler_D_noise = lr_scheduler.StepLR(
            self.optimizerD_noise, 
            step_size=opt.lr_decay_iters_D, 
            gamma=0.1)
        self.scheduler_D_denoise = lr_scheduler.StepLR(
            self.optimizerD_denoise, 
            step_size=opt.lr_decay_iters_D, 
            gamma=0.1)
        self.scheduler_Dl = lr_scheduler.StepLR(
            self.optimizerD_latent, 
            step_size=opt.lr_decay_iters_D, 
            gamma=0.1)

        # Lists to keep track of progress
        self.losses_adv = {}
        self.losses = {}
        self.n_iters = opt.n_iters
        self.weight_grad_adv = opt.weight_grad_adv
        self.weight_adv = opt.weight_adv
        self.weight_adv_latent = opt.weight_adv_latent
        self.weight_mmd = opt.weight_mmd
        self.weight_cycle = opt.weight_cycle
        self.opt = opt
        self.weight_sides = opt.weight_sides

    def training(
        self,
        data_loader, 
        data_loader_val,
        num_epochs=1,
        loss='l2',
        burnin=-1,
        iters_freq=500,
        save_freq=5,
        incr_step = 10
        ):        
        print("Starting Training Loop...")

        if loss == 'l2':
            criterionRec = torch.nn.MSELoss(reduction='mean')
        else:
            criterionRec = torch.nn.L1Loss(reduction='mean')

        dataiter = iter(data_loader_val)
        syn_tgt_test, real_tgt_test, _ = dataiter.next()
        syn_tgt_test, real_tgt_test =\
            syn_tgt_test.to(self.device), real_tgt_test.to(self.device)

        writer = SummaryWriter()
        iters = 0
        try:
            for epoch in range(num_epochs):
                for syn_tgt, real_tgt, _ in data_loader:                                                    
                    signal_syn, signal_real = signal_to_device(
                        syn_tgt[:, :, :, 64:-64],
                        real_tgt[:, :, :, 64:-64], self.device)

                    left_syn, right_syn =\
                        signal_to_device(
                            syn_tgt[:, :, :, :64], 
                            syn_tgt[:, :, :, -64:], self.device)

                    left_real, right_real =\
                        signal_to_device(
                            real_tgt[:, :, :, :64], 
                            real_tgt[:, :, :, -64:], self.device) 
                        
                    # train classifiers
                    if (iters % self.n_iters == 0):
                        self.netD_denoise.zero_grad()
                        with torch.no_grad():
                            with_style = self.StyleNet(signal_real, real=False)

                        err, acc, adv = classifier_step(
                            self.netD_denoise, 
                            with_style, signal_syn,
                            self.weight_grad_adv, self.device)
                        err.backward()

                        self.optimizerD_denoise.step()
                        self.losses_adv['DISC_denoise'] = err.item()
                        self.losses_adv['Accuracy_DISC_denoise'] = acc
                        self.losses_adv['DISC_adversary_penalty_denoise'] = adv

                        self.netD_noise.zero_grad()
                        with torch.no_grad():
                            with_style = self.StyleNet(signal_syn, real=True)
                        err, acc, adv = classifier_step(
                            self.netD_noise, 
                            with_style, signal_real,
                            self.weight_grad_adv, self.device)
                        err.backward()
                        self.optimizerD_noise.step()
                        self.losses_adv['DISC_noise'] = err.item()
                        self.losses_adv['Accuracy_DISC_noise'] = acc
                        self.losses_adv['DISC_adversary_penalty_noise'] = adv

                        self.netD_latent.zero_grad()
                        with torch.no_grad():               
                            latent_real = self.StyleNet.encode(signal_real)
                            latent_syn = self.StyleNet.encode(signal_syn)

                        err_latent, acc, adv = classifier_step(
                            self.netD_latent, 
                            latent_syn, latent_real,
                            self.weight_grad_adv, self.device)
                        err_latent.backward()
                        self.optimizerD_latent.step()
                        self.losses_adv['DISC_latent'] = err_latent.item()
                        self.losses_adv['Accuracy_DISC_latent'] = acc                        
                        self.losses_adv['DISC_adversary_penalty_latent'] = adv
                    else:
                        for k in self.losses_adv.keys():
                            self.losses_adv[k] = 0                    

                    # train network
                    self.StyleNet.train()
                    self.StyleNet.zero_grad()
                    self.netD_noise.zero_grad()
                    self.netD_denoise.zero_grad()
                    self.netD_latent.zero_grad()

                    # compute latent representations. we will enforce them to be 
                    # indestinguishable
                    latent_real = self.StyleNet.encode(signal_real)
                    latent_syn = self.StyleNet.encode(signal_syn)

                    self.losses['latent_real_mean'] = latent_real.clone().detach().mean()
                    self.losses['latent_real_std'] = latent_real.clone().detach().std()
                    self.losses['latent_synl_mean'] = latent_syn.clone().detach().mean()
                    self.losses['latent_synl_std'] = latent_syn.clone().detach().std()

                    # adversarial latent loss
                    b_size = signal_real.size(0)
                    label_real = torch.full(
                        (b_size,), real_label, dtype=torch.float, 
                        device=self.device)
                    label_syn = torch.full(
                        (b_size,), syn_label, dtype=torch.float, 
                        device=self.device)

                    outputDl_syn = self.netD_latent(latent_syn).view(-1)
                    outputDl_real = self.netD_latent(latent_real).view(-1)
                    errD_latent = criterion(outputDl_syn, label_real) +\
                        criterion(outputDl_real, label_syn)
                    acc = get_accuracy(torch.cat((outputDl_syn, outputDl_real), dim=0),\
                        torch.cat((label_real, label_syn), dim=0), self.device)
                    self.losses['Accuracy_ADV_latent'] = acc

                    # latent feature matching with mmd
                    loss_fm = FeatureMatching(latent_real, latent_syn)
                                        
                    rec_real = self.StyleNet.decode(latent_real, real=True)
                    rec_real_assyn = self.StyleNet.decode(latent_real, real=False)
                    rec_syn = self.StyleNet.decode(latent_syn, real=False)                    
                    rec_syn_asreal = self.StyleNet.decode(latent_syn, real=True)
                    
                    # simple reconstruction loss
                    err_MSE = criterionRec(rec_syn, signal_syn) +\
                        criterionRec(rec_real, signal_real)

                    # add cycle consistency loss
                    rec_syn_cycle = self.StyleNet(rec_syn_asreal, real=False)
                    rec_real_cycle = self.StyleNet(rec_real_assyn, real=True)

                    err_cycle = criterionRec(rec_syn_cycle, signal_syn) +\
                        criterionRec(rec_real_cycle, signal_real)

                    outputD_denoise = self.netD_denoise(rec_real_assyn).view(-1)
                    b_size = outputD_denoise.size(0)
                    label_real = torch.full(
                        (b_size,), real_label, dtype=torch.float, device=self.device)
                    label_syn = torch.full(
                        (b_size,), syn_label, dtype=torch.float, device=self.device)

                    err_D_denoise = criterion(outputD_denoise, label_real)
                    acc = get_accuracy(outputD_denoise, label_real, self.device)
                    self.losses['Accuracy_ADV_denoise'] = acc

                    outputD_noise = self.netD_noise(rec_syn_asreal).view(-1)
                    err_D_noise = criterion(outputD_noise, label_real)
                    acc = get_accuracy(outputD_noise, label_real, self.device)
                    self.losses['Accuracy_ADV_noise'] = acc

                    reconsts_left = self.StyleNet.decode(
                        latent_syn, sides='left')
                    reconsts_right = self.StyleNet.decode(
                        latent_syn, sides='right')

                    reconsts_left_real = self.StyleNet.decode(
                        latent_real, sides='left')
                    reconsts_right_real = self.StyleNet.decode(
                        latent_real, sides='right')

                    rec_loss_right = criterionRec(reconsts_right, right_syn)
                    rec_loss_left = criterionRec(reconsts_left, left_syn)

                    self.losses['SIDES_right'] = rec_loss_right.item()
                    self.losses['SIDES_left'] = rec_loss_left.item()

                    loss_total = err_MSE +\
                        self.weight_sides*(rec_loss_right + rec_loss_left) +\
                        self.weight_cycle * err_cycle +\
                        self.weight_mmd*loss_fm +\
                        self.weight_adv_latent*errD_latent

                    if iters > burnin:
                        loss_total +=\
                            self.weight_adv*(err_D_denoise + err_D_noise)                            

                    self.losses['LATENT_mmd'] = loss_fm.item()
                    self.losses['REC_mse'] = err_MSE.item()
                    self.losses['REC_cycle'] = err_cycle.item()
                    self.losses['ADV_noise'] = err_D_noise.item()
                    self.losses['ADV_denoise'] = err_D_denoise.item()
                    self.losses['ADV_latent'] = errD_latent.item()

                    self.losses['TOTAL_LOSS'] = loss_total.item()

                    self.losses['Variance'] =\
                        torch.exp(self.StyleNet.logvar.weight[0]).max()

                    loss_total.backward()
                    self.optimizerS.step()                                        
                    iters += 1

                writer.add_image('INPUT Synthetic',
                    pretty_batch(signal_syn), epoch)
                writer.add_image('INPUT Real',
                    pretty_batch(signal_real), epoch) 
                writer.add_image('RECONSTRUCTION Real', 
                    pretty_batch(rec_real), epoch)
                writer.add_image('RECONSTRUCTION Synthetic',
                    pretty_batch(
                        torch.cat((reconsts_left,
                            rec_syn,
                            reconsts_right), 3)
                        ), epoch)
                writer.add_image('RECONSTRUCTION Real to syn',
                    pretty_batch(
                        torch.cat((reconsts_left_real,
                            rec_real_assyn,
                            reconsts_right_real), 3)
                        ), epoch)                
                writer.add_image('RECONSTRUCTION Synthetic to real',
                    pretty_batch(rec_syn_asreal), epoch)
                writer.add_image('RECONSTRUCTION Real (Cycle)',
                    pretty_batch(rec_real_cycle), epoch)
                writer.add_image('RECONSTRUCTION Synthetic (Cycle)',
                    pretty_batch(rec_syn_cycle), epoch)                        
                    
                print(f"[{epoch}/{num_epochs}][{iters}]\t",\
                    f"Rec: {self.losses['REC_mse']:.2e}",
                    f"Acc latent: {self.losses['Accuracy_ADV_latent']:.2f}",
                    f"Acc denoise: {self.losses['Accuracy_ADV_denoise']:.2f}",
                    f"Acc noise: {self.losses['Accuracy_ADV_noise']:.2f}")

                self.StyleNet.eval()
                with torch.no_grad():
                    syn_gen = self.StyleNet(
                        syn_tgt_test[:, :, :, 64:-64], 
                        real=False, full=True)
                    real_gen = self.StyleNet(
                        real_tgt_test[:, :, :, 64:-64],
                        real=True, full=False).detach()
                    syn_to_real = self.StyleNet(
                        syn_tgt_test[:, :, :, 64:-64],
                        real=True, full=False).detach()
                    real_to_syn = self.StyleNet(
                        real_tgt_test[:, :, :, 64:-64],
                        real=False, full=True).detach()

                    self.losses['SIDES_right_VAL'] = criterionRec(
                        syn_gen[:, :, :, -64:], 
                        syn_tgt_test[:, :, :, -64:]).item()
                    self.losses['SIDES_left_VAL'] = criterionRec(
                        syn_gen[:, :, :, :64], 
                        syn_tgt_test[:, :, :, :64]).item()

                writer.add_image('VAL TARGET Synthetic', 
                    pretty_batch(syn_tgt_test), epoch)
                writer.add_image('VAL TARGET Real',
                    pretty_batch(real_tgt_test), epoch) 
                writer.add_image('VAL RECONSTRUCTION Real',
                    pretty_batch(real_gen), epoch)
                writer.add_image('VAL RECONSTRUCTION Synthetic',
                    pretty_batch(syn_gen), epoch)
                writer.add_image('VAL RECONSTRUCTION Real to syn',
                    pretty_batch(real_to_syn), epoch)
                writer.add_image('VAL RECONSTRUCTION Synthetic to real',
                    pretty_batch(syn_to_real), epoch)

                # evaluations
                for loss_name in self.losses.keys():
                    writer.add_scalar(f'Full/{loss_name}',
                    self.losses[loss_name], epoch)

                for loss_name in self.losses_adv.keys():
                    writer.add_scalar(f'Full/{loss_name}',
                    self.losses_adv[loss_name], epoch)

                self.StyleNet.train()                                        

                
                if ((epoch % incr_step) == 0) and (self.weight_adv_latent < 1) and (epoch > 0):
                    self.weight_adv_latent *= 5
                    print(f'Increased weight_adv_latent to: ', 
                        self.weight_adv_latent)
                if ((epoch % incr_step) == 0) and (self.weight_adv < 1) and (epoch > 0):
                    self.weight_adv *= 5
                    print(f'Increased weight_adv to: ', 
                        self.weight_adv)
                
                self.scheduler_D_noise.step()
                self.scheduler_D_denoise.step()
                self.scheduler_Dl.step()
                self.scheduler_G.step()

                if (epoch % save_freq == 0) and (epoch > 0):
                    print(f'Autosave epoch = {epoch}')
                    self.save(suffix=str(epoch))
            
            writer.flush()
            writer.close()
            self.save()

        except KeyboardInterrupt:
            writer.flush()
            writer.close()
            self.save()

    def save(self, suffix=''):
        torch.save(self.StyleNet.state_dict(), f'{self.logfile_name}/Style{suffix}.pth')
        torch.save(self.netD_latent.state_dict(), f'{self.logfile_name}/netD_latent{suffix}.pth')
        torch.save(self.netD_denoise.state_dict(), f'{self.logfile_name}/netD_denoise{suffix}.pth')
        torch.save(self.netD_noise.state_dict(), f'{self.logfile_name}/netD_noise{suffix}.pth')
        print(f'Results saved to {self.logfile_name}')