from asyncio import tasks
import torch
from torch.optim import lr_scheduler

import numpy

from utils import *
# from dataLoader import *
from model import *

from pytorch_lightning import LightningModule
import math
import wandb
from PIL import Image


class sidesModel(LightningModule):
    """Base model with training loop, loss, and test configuration."""

    def __init__(
            self,
            weight_sides=10.0,
            pretrained_style=None,
            loss='l1',
            burnin=100,

            warmup=40,
            max_iters=1000,
            writer='tmp',
            epochs=1,
            learning_rate=1e-3,
            parallelism=1,
            print_freq=1,
            logfile='tmp.txt',
            num_epochs=1,
            **kwargs
    ):
        super().__init__()
        if kwargs:
            print("Ignoring extra kwargs: %s" % kwargs)

        self.save_hyperparameters()
        if loss == 'l2':
            self.loss = torch.nn.MSELoss(reduction='mean')
        else:
            self.loss = torch.nn.L1Loss(reduction='mean')

        self._create_model()

    def _create_model(self):
        if not (self.hparams.pretrained_style is None):
            self.StyleNet = StyleNetwork(-1)
            self.StyleNet.load_state_dict(
                torch.load(f'{self.hparams.pretrained_style}/Style.pth'))
            self.StyleNet.eval()

        self.SidesNet = FaderNetwork()
        
        self.weight_sides = self.hparams.weight_sides

    def forward(self, syn_assyn, real_assyn=None):
        """
        Args:

        """

        # create necessary constansts
        reconsts_syn = self.SidesNet(syn_assyn)

        if real_assyn is None:
            reconsts_real = None
        else:
            reconsts_real = self.SidesNet(real_assyn)

        return reconsts_syn, reconsts_real
         
    def training_step(self, batch, batch_idx):
        self.log('lr', self.lr_scheduler.get_last_lr()[0], prog_bar=True)
        _, _, loss = self._step(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, loss = self._step(batch, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        syn_tgt_test = batch
        reconsts_syn, reconsts_real, loss = self._step(batch, 'test')
        return reconsts_syn, reconsts_real, loss

    def _step(self, batch, loss_name):
        # if self.trainer.global_step == ( self.total_steps // 2):
        if self.current_epoch == (self.hparams.num_epochs//2):
            self.loss = torch.nn.L1Loss(reduction='mean')
            print('Switched to l1 loss')

        syn_tgt = batch

        if not (self.hparams.pretrained_style is None):
            with torch.no_grad():
                syn_assyn =\
                    self.StyleNet(
                        syn_tgt[:, :, :, 64:-64], real=False
                        ).clone().detach()                    
                # real_assyn =\
                #     self.StyleNet(
                #         real_tgt[:, :, :, 64:-64], real=False
                #         ).clone().detach()
        else:
            syn_assyn = syn_tgt[:, :, :, 64:-64]
            # real_assyn = real_tgt[:, :, :, 64:-64]

        # reconsts_syn, reconsts_real = self.forward(syn_assyn, real_assyn)
        reconsts_syn, reconsts_real = self.forward(syn_assyn, None)
                    
        rec_loss_sides =\
            self.loss(reconsts_syn[:, :, :, -64:], 
            syn_tgt[:, :, :, -64:]) +\
            self.loss(reconsts_syn[:, :, :, :64], 
            syn_tgt[:, :, :, :64])

        self.log(f'LossSides/{loss_name}', rec_loss_sides.item(), prog_bar=True)

        if self.current_epoch > self.hparams.burnin:
            loss =\
                self.weight_sides*rec_loss_sides
        else:
            rec_loss_center = self.loss(reconsts_syn[:, :, :, 64:-64], syn_assyn)
            # rec_loss_real = self.loss(reconsts_real[:, :, :, 64:-64], real_assyn)
            loss =\
                self.weight_sides*rec_loss_sides +\
                    rec_loss_center 
            self.log(f'LossCenter/{loss_name}', rec_loss_center.item(), prog_bar=True)
            # + rec_loss_real

        self.log(f'Loss/{loss_name}', loss.item(), prog_bar=True)
        

        if loss_name != 'train':
            im = Image.fromarray(np.uint8(cm.gist_earth(syn_tgt[0][0].detach().cpu().numpy())*255))
            images = wandb.Image(im, caption="Target Synthetic")
            wandb.log({f"Target Synthetic ({loss_name})": images})

            im = Image.fromarray(np.uint8(cm.gist_earth(reconsts_syn[0][0].detach().cpu().numpy())*255))
            images = wandb.Image(im, caption="OUTPUT Synthetic")
            wandb.log({f"OUTPUT Synthetic ({loss_name})": images})

        # im = Image.fromarray(np.uint8(cm.gist_earth(reconsts_syn[0][0][:, :64].detach().cpu().numpy())*255))
        # images = wandb.Image(im, caption="OUTPUT Synthetic")
        # wandb.log({f"OUTPUT Left Synthetic ({loss_name})": images})

        # im = Image.fromarray(np.uint8(cm.gist_earth(reconsts_syn[0][0][:, -64:].detach().cpu().numpy())*255))
        # images = wandb.Image(im, caption="OUTPUT Synthetic")
        # wandb.log({f"OUTPUT Right Synthetic ({loss_name})": images})
        
        # self.writer.add_image('TARGET Synthetic',
        #     pretty_batch(syn_tgt), self.current_epoch)
        # self.writer.add_image('TARGET Real',
        #     pretty_batch(real_tgt), self.current_epoch)

        # self.writer.add_image('OUTPUT Synthetic',
        #     pretty_batch(reconsts_syn.detach()), self.current_epoch)
        # self.writer.add_image('OUTPUT Real',
        #     pretty_batch(reconsts_real.detach()), self.current_epoch)

        return reconsts_syn, reconsts_real, loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=1e-5)
            # betas=(0.9, 0.98))
            # , eps=1e-9)

        # self.lr_scheduler =\
        #     pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
        #         optimizer, warmup_epochs=10, max_epochs=self.hparams.max_iters)
        
        self.total_steps = self.trainer.datamodule.train_dataloader_len//\
            self.trainer.datamodule.batch_size*self.hparams.num_epochs
        
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, 
                T_max=self.total_steps, 
                eta_min=1e-7)
        
        sched = {
            'scheduler': self.lr_scheduler,
            'interval': 'step',
        }

        return [optimizer], [sched]


