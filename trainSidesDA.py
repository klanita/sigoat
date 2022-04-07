import torch
from torch.optim import lr_scheduler

import numpy
from utils import *
# from dataLoader import *
from model import *
from torch.utils.tensorboard import SummaryWriter


class TrainerSides:
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

        print('Pretrained style:', opt.pretrained_style)
        self.StyleNet = StyleNetwork(-1).to(device)
        if not (opt.pretrained_style is None):
            self.StyleNet.load_state_dict(
                torch.load(f'{opt.pretrained_style}/Style.pth',
                map_location=torch.device(device)))
        self.StyleNet.eval()        

        print('Pretrained sides:', opt.pretrained)
        if not ((opt.pretrained is None) or (opt.pretrained == 'None')):
            self.SidesNet = FaderNetwork().to(device)
            self.SidesNet.load_state_dict(
                torch.load(opt.pretrained + '/sides.pth',
                map_location=torch.device(device)))
        else:
            self.SidesNet = FaderNetwork().to(device)
            # self.SidesNet = FaderNetwork(normalization='instance').to(device)
            # self.SidesNet.encoder.load_state_dict(self.StyleNet.encoder.state_dict())
            
        self.SidesNet.train()

        self.optimizer = optim.Adam(
            list(self.SidesNet.parameters()),
            lr=opt.lr,
            betas=(opt.beta1, 0.999))

        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=10, 
            gamma=0.1)

        self.sides_weight = 1
        self.losses = {}
        self.weight_sides = opt.weight_sides

    def training(
        self, 
        data_loader, 
        data_loader_val, 
        num_epochs=1, 
        loss='l1', 
        burnin=1000
        ):
        print(f"Starting Training Loop with {loss} for {num_epochs} epochs...")

        if loss == 'l2':
            loss = torch.nn.MSELoss(reduction='mean')
        else:
            loss = torch.nn.L1Loss(reduction='mean')
        
        dataiter = iter(data_loader_val)
        syn_tgt_test, real_tgt_test, _ = dataiter.next()
        syn_tgt_test = syn_tgt_test.to(self.device)
        real_tgt_test = real_tgt_test.to(self.device)

        with torch.no_grad():
            syn_assyn_test = self.StyleNet(
                syn_tgt_test[:, :, :, 64:-64], 
                real=False)
            real_assyn_test = self.StyleNet(
                real_tgt_test[:, :, :, 64:-64],
                real=False)

        writer = SummaryWriter()
        iters = 0
        try:
            for epoch in range(num_epochs):
                for syn_tgt, real_tgt, _ in data_loader:
                    syn_tgt, real_tgt =\
                        signal_to_device(syn_tgt, real_tgt, self.device)
 
                    with torch.no_grad():
                        syn_assyn =\
                            self.StyleNet(
                                syn_tgt[:, :, :, 64:-64], real=False
                                ).clone().detach()                    
                        real_assyn =\
                            self.StyleNet(
                                real_tgt[:, :, :, 64:-64], real=False
                                ).clone().detach()
                    
                    self.SidesNet.train()
                    self.SidesNet.zero_grad()

                    reconsts_syn = self.SidesNet(syn_assyn)
                    reconsts_real = self.SidesNet(real_assyn)

                    rec_loss =\
                        loss(reconsts_syn[:, :, :, 64:-64], syn_assyn)
                    rec_loss_real =\
                        loss(reconsts_real[:, :, :, 64:-64], real_assyn)
                    
                    rec_loss_right =\
                        loss(reconsts_syn[:, :, :, -64:], 
                        syn_tgt[:, :, :, -64:])
                    rec_loss_left =\
                        loss(reconsts_syn[:, :, :, :64], 
                        syn_tgt[:, :, :, :64])

                    if iters > burnin:
                        total_loss =\
                            self.weight_sides*(rec_loss_right + rec_loss_left)
                    else:
                        total_loss =\
                            self.weight_sides*(rec_loss_right + rec_loss_left)+\
                                rec_loss + rec_loss_real

                    total_loss.backward()
                    self.optimizer.step()

                    self.losses['TRAIN_center'] = rec_loss.item()
                    self.losses['TRAIN_left'] = rec_loss_left.item()
                    self.losses['TRAIN_right'] = rec_loss_right.item()
                    self.losses['TRAIN_center_REAL'] = rec_loss_real.item()
                    self.losses['TOTAL_LOSS'] = total_loss.item()
                    
                    if iters % 100 == 0:
                        with torch.no_grad():
                            self.SidesNet.eval()
                            reconsts_syn_test = self.SidesNet(syn_assyn_test)
                            self.losses['VAL_center'] =\
                                loss(reconsts_syn_test[:, :, :, 64:-64], 
                                syn_assyn_test).item()
                            self.losses['VAL_left'] =\
                                loss(reconsts_syn_test[:, :, :, :64], 
                                syn_tgt_test[:, :, :, :64]).item()
                            self.losses['VAL_right'] =\
                                loss(reconsts_syn_test[:, :, :, -64:], 
                                syn_tgt_test[:, :, :, -64:]).item()

                            reconsts_real_test =\
                                self.SidesNet(real_assyn_test)
                            self.losses['VAL_center_real'] =\
                                loss(reconsts_real_test[:, :, :, 64:-64], 
                                real_assyn_test).item()
                            self.losses['VAL_left_real'] =\
                                loss(reconsts_real_test[:, :, :, :64], 
                                real_tgt_test[:, :, :, :64]).item()
                            self.losses['VAL_right_real'] =\
                                loss(reconsts_real_test[:, :, :, -64:], 
                                real_tgt_test[:, :, :, -64:]).item()

                            self.SidesNet.train()

                        for loss_name in self.losses.keys():
                            writer.add_scalar(f'Sides/{loss_name}',
                            self.losses[loss_name], iters)
                    
                        writer.add_image('TARGET Synthetic',
                            pretty_batch(syn_tgt), iters)
                        writer.add_image('TARGET Real',
                            pretty_batch(real_tgt), iters)
                            
                        writer.add_image('INPUT Synthetic',
                            pretty_batch(syn_assyn), iters)
                        writer.add_image('INPUT Real',
                            pretty_batch(real_assyn), iters)

                        writer.add_image('RECONSTRUCTION Synthetic',
                            pretty_batch(reconsts_syn), iters)
                        writer.add_image('RECONSTRUCTION Real',
                            pretty_batch(reconsts_real), iters)

                        writer.add_image('VAL TARGET Real',
                            pretty_batch(real_assyn_test), iters)
                        writer.add_image('VAL TARGET Synthetic',
                            pretty_batch(syn_assyn_test), iters) 
                        writer.add_image('VAL RECONSTRUCTION Synthetic',
                            pretty_batch(reconsts_syn_test), iters)
                        writer.add_image('VAL RECONSTRUCTION Real',
                            pretty_batch(reconsts_real_test), iters)                         

                        print(f"[{epoch}/{num_epochs}][{iters}]\t",\
                                f"Total_loss: {self.losses['TOTAL_LOSS']:.2e}")

                    iters += 1
                self.scheduler.step()
                if (epoch % 5 == 0) and (epoch > 0):
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
        torch.save(self.SidesNet.state_dict(), f'{self.logfile_name}/sides{suffix}.pth')
        print(f'Results saved to {self.logfile_name}')


        