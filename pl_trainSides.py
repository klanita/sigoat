from asyncio import tasks
import torch
from torch.optim import lr_scheduler

import numpy

from utils import *
# from dataLoader import *
from model import *

from pytorch_lightning import LightningModule
import math
import pl_bolts
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
        
        self.sides_weight = 1
        self.weight_sides = self.hparams.weight_sides

    def forward(self, syn_assyn, real_assyn):
        """
        Args:

        """

        # create necessary constansts
        reconsts_syn = self.SidesNet(syn_assyn)
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
        syn_tgt_test, real_tgt_test, _ = batch
        reconsts_syn, reconsts_real, loss = self._step(batch, 'test')
        return reconsts_syn, reconsts_real, loss

    def _step(self, batch, loss_name):
        syn_tgt, real_tgt, _ = batch

        if not (self.hparams.pretrained_style is None):
            with torch.no_grad():
                syn_assyn =\
                    self.StyleNet(
                        syn_tgt[:, :, :, 64:-64], real=False
                        ).clone().detach()                    
                real_assyn =\
                    self.StyleNet(
                        real_tgt[:, :, :, 64:-64], real=False
                        ).clone().detach()
        else:
            syn_assyn = syn_tgt[:, :, :, 64:-64]
            real_assyn = real_tgt[:, :, :, 64:-64]

        reconsts_syn, reconsts_real = self.forward(syn_assyn, real_assyn)
                    
        rec_loss_right =\
            self.loss(reconsts_syn[:, :, :, -64:], 
            syn_tgt[:, :, :, -64:])
        rec_loss_left =\
            self.loss(reconsts_syn[:, :, :, :64], 
            syn_tgt[:, :, :, :64])

        if self.current_epoch > self.hparams.burnin:
            loss =\
                self.weight_sides*(rec_loss_right + rec_loss_left)
        else:
            rec_loss = self.loss(reconsts_syn[:, :, :, 64:-64], syn_assyn)
            rec_loss_real = self.loss(reconsts_real[:, :, :, 64:-64], real_assyn)
            loss =\
                self.weight_sides*(rec_loss_right + rec_loss_left)+\
                    rec_loss + rec_loss_real

        self.log(f'Loss/{loss_name}', loss.item(), prog_bar=True)

        im = Image.fromarray(pretty_batch(reconsts_syn.detach()))
        images = wandb.Image(im, caption="TARGET Synthetic")
        wandb.log({f"OUTPUT Synthetic": images})
        
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
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            betas=(0.9, 0.98), eps=1e-9)

        self.lr_scheduler =\
            pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=10, max_epochs=self.hparams.max_iters)
        sched = {
            'scheduler': self.lr_scheduler,
            'interval': 'step',
        }

        return [optimizer], [sched]


