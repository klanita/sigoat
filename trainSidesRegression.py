import torch
from torch.optim import lr_scheduler

import numpy

from utils import *
# from dataLoader import *
from model import *
from torch.utils.tensorboard import SummaryWriter

class TrainerReconstruction:
    def __init__(
        self,
        opt,
        logfile_name,
        device=None
        ):
        self.logfile_name = logfile_name
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.StyleNet = StyleNetwork(-1).to(device)
        if not (opt.pretrained_style is None):
            self.StyleNet.load_state_dict(
                torch.load(f'{opt.pretrained_style}/Style.pth',
                map_location=torch.device(device)))
        self.StyleNet.eval()        
        
        self.RecNet = ReconstructionNetwork().to(device)
        print('Loading pretrained model:', opt.pretrained)
        if not ((opt.pretrained is None) or (opt.pretrained == 'None')):
            print('Loading pretrained model:', opt.pretrained)
            self.RecNet.load_state_dict(
                torch.load(f"{opt.pretrained}/Reconstruction_{opt.split}.pth",
                map_location=torch.device(device)))
            
        self.RecNet.train()

        self.optimizer = optim.Adam(
            list(self.RecNet.parameters()),
            lr=opt.lr,
            betas=(opt.beta1, 0.999))

        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=opt.lr_decay_iters, 
            gamma=0.1)

        self.sides_weight = 1
        self.losses = {}
        self.split = opt.split
        print(f"Training {opt.split} side")
        self.center_idx = range(64, 256-64)
        if opt.split == 'left':
            self.split_idx = range(0, 128)
        elif opt.split == 'right':
            self.split_idx = range(128, 256)
        else:
            raise NotImplementedError

    def training(
        self, 
        data_loader, 
        data_loader_val, 
        num_epochs=1, 
        loss='l2'
        ):

        print("Starting Training Loop...")

        if loss == 'l2':
            loss = torch.nn.MSELoss(reduction='mean')
        else:
            loss = torch.nn.L1Loss(reduction='mean')
        
        dataiter = iter(data_loader_val)
        signal_syn_tgt_test, signal_real_tgt_test, _ = dataiter.next()
        signal_syn_test, signal_real_test =\
            signal_to_device(signal_syn_tgt_test[:, :, :, self.center_idx], 
            signal_real_tgt_test[:, :, :, self.center_idx], self.device)
        signal_syn_tgt_test, signal_real_tgt_test =\
            signal_to_device(signal_syn_tgt_test[:, :, :, self.split_idx], 
            signal_real_tgt_test[:, :, :, self.split_idx], self.device)

        with torch.no_grad():
            syn_to_real_test = self.StyleNet(
                signal_syn_test, real=False).clone().detach()

        writer = SummaryWriter()
        iters = 0
        try:
            for epoch in range(num_epochs):
                for signal_syn_tgt, signal_real_tgt, _ in data_loader:
                    signal_syn,  signal_syn_tgt = signal_to_device(
                        signal_syn_tgt[:, :, :, self.center_idx],
                        signal_syn_tgt[:, :, :, self.split_idx], self.device)                                    
                    signal_real, signal_real_tgt = signal_to_device(
                        signal_real_tgt[:, :, :, self.center_idx],
                        signal_real_tgt[:, :, :, self.split_idx], self.device)                  

                    with torch.no_grad():
                        syn_to_real = self.StyleNet(signal_syn, real=False).clone().detach()                    
                    
                    self.RecNet.train()
                    self.RecNet.zero_grad()

                    reconsts = self.RecNet(syn_to_real)
                    total_loss = loss(reconsts, signal_syn_tgt)

                    total_loss.backward()
                    self.optimizer.step()
                    self.losses['Total_loss'] = total_loss.item()                                    
                    iters += 1

                print(f"[{epoch}/{num_epochs}][{iters}]\t",\
                    f"Total_loss: {self.losses['Total_loss']:.2e}")

                with torch.no_grad():
                    self.RecNet.eval()                            
                    rec_real = self.StyleNet(signal_real, real=False).clone().detach()
                    reconsts_real = self.RecNet(rec_real.clone().detach())
                    rec_loss_real = loss(reconsts_real, signal_real_tgt)
                    self.losses['Rec_real'] = rec_loss_real.item()

                    reconsts_syn_test = self.RecNet(syn_to_real_test.clone().detach())
                    rec_loss_test = loss(reconsts_syn_test, signal_syn_tgt_test)
                    self.losses['VAL_syn'] = rec_loss_test.item()

                    self.RecNet.train()

                for loss_name in self.losses.keys():
                    writer.add_scalar(f'Reconstruction/{loss_name}',
                    self.losses[loss_name], epoch)

                writer.add_image('INPUT Synthetic',
                    pretty_batch(signal_syn), epoch)
                writer.add_image('TARGET Synthetic',
                    pretty_batch(signal_syn_tgt), epoch)
                writer.add_image('RECONSTRUCTION Synthetic to real',
                    pretty_batch(syn_to_real), epoch)
                writer.add_image('INPUT Real',
                    pretty_batch(signal_real), epoch) 
                writer.add_image('RECONSTRUCTION Real to syn',
                    pretty_batch(rec_real), epoch)
                writer.add_image('RECONSTRUCTION Real',
                    pretty_batch(reconsts_real), epoch)
                writer.add_image('RECONSTRUCTION Synthetic',
                    pretty_batch(reconsts), epoch)                    
                
                if (epoch > 0) and (epoch % 5 == 0):
                    self.save()

            writer.flush()
            writer.close()
            self.save()

        except KeyboardInterrupt:
            writer.flush()
            writer.close()
            self.save()

    def save(self):
        torch.save(self.RecNet.state_dict(), f'{self.logfile_name}/Reconstruction_{self.split}.pth')
        print(f'Results saved to {self.logfile_name}')


        