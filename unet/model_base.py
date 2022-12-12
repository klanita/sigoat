import pytorch_lightning as pl
import torch
import numpy as np

from torch.nn import functional as F
# from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef

import pl_bolts

import logging
import torch
import numpy as np

from torch.nn import functional as F
from torch.nn.functional import mse_loss, l1_loss
import wandb
from PIL import Image
from matplotlib import cm

from sklearn.metrics import mean_squared_error, mean_absolute_error
import h5py

out_logger = logging.getLogger(__name__)

""" Template DeepXIC Module to be inherited"""
class Parent_Model(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        if kwargs: 
            out_logger.warning("Ignoring extra kwargs: %s" % kwargs)

        self.save_hyperparameters() #makes hparams available

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
                                self.parameters(), 
                                lr=self.hparams.learning_rate, 
                                betas=(0.9, 0.98), 
                                weight_decay=self.hparams.weight_decay)

        self.lr_scheduler = {}

        if self.hparams.scheduler == "COSINE":
            total_steps = len(self.trainer.datamodule.train_set)//self.hparams.batch_size*self.hparams.num_epochs

            sch = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=self.hparams.cosine_warmup_epochs, max_epochs=total_steps)

            self.lr_scheduler = {'scheduler' : sch,
                         'name' : 'learning rate',
                         'interval' : 'step',
                         'frequency' : 1}
                         
        elif self.hparams.scheduler == "STEPLR" :
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, 
                factor=self.hparams.steplr_factor, 
                mode='min', 
                patience=self.hparams.steplr_patience)

            self.lr_scheduler = {"scheduler": sch, 
                         "name": 'learning rate',
                         "interval": 'epoch',
                         'frequency': 1,
                         "monitor": "Loss/val"}

        elif self.hparams.scheduler == 'CONST' :
            return [optimizer]

        else :
            out_logger.critical("scheduler could not be initialized")

        return [optimizer], [self.lr_scheduler]

    def _step(self, batch, loss_name) :

        batch_input, target = batch

        output, _ = self.forward(batch_input)
        loss = self._loss(output, target)

        if loss_name != 'test' :
            self.log(f'Loss/{loss_name}', loss.item(), prog_bar=True)

        return output, target, loss

    def training_step(self, batch, batch_idx):
        *_, loss = self._step(batch, 'train')

        return loss

    def validation_step(self, batch, batch_idx):
        output, target, loss = self._step(batch, 'val')
        return {"output" : output,
                "target" : target}

    def validation_step_end(self, batch) :
        metrics = self._metrics(batch['output'].cpu(),
                                batch['target'].cpu())
        for k in metrics :
            self.log(f'{k}/val', metrics[k], prog_bar=True)

    def test_step(self, batch, batch_idx):
        output, target, loss = self._step(batch, 'test')

        return output, target, loss
    
    def test_step_end(self, test_step_output):
        output, target, *_ = test_step_output
        
        loss = self._loss(output , target)
        self.log("Loss/test", loss.item())

        metrics = self._metrics(output , target)
        for m in metrics:
            self.log(f'{m}/test', metrics[m], prog_bar=False)

    def test_epoch_end(self, test_output):

        all_output = []
        all_target = []
        all_loss = []
        for batch in test_output:
            output, target, loss = batch
            all_output.append(output)
            all_target.append(target)
            all_loss.append(loss)

        all_output = torch.cat(all_output)
        all_target = torch.cat(all_target)
        all_loss = torch.cat(all_loss)

        metrics = self._metrics(all_output.cpu(), 
                                all_target.cpu())

        for k in metrics:
            self.log(f'{k}/test_final', metrics[k], prog_bar=True)


""" Template DeepXIC Module to be inherited"""
class AutoEncoderModel(Parent_Model) :
    def __init__(self, **kwargs):
        super().__init__()

    def _loss(self, img_pred, img_tgt):
        if self.hparams.loss_name == 'l2':
            return mse_loss(img_pred.reshape(-1), img_tgt.reshape(-1))
        elif self.hparams.loss_name == 'l1':
            return l1_loss(img_pred.reshape(-1), img_tgt.reshape(-1))
        else:
            raise NotImplementedError

    def _metrics(self, pred, target):

        if torch.is_tensor(pred) :
            pred, target = pred.cpu(), target.cpu()

        metrics = {}
        pred, target = pred.squeeze(1).cpu(), target.cpu()

        metrics['MSE'] = mean_squared_error(y_true=target, y_pred=pred)
        metrics['MAE'] = mean_absolute_error(y_true=target, y_pred=pred)
        metrics['RMAE'] = metrics['MAE']/(max(target) - min(target))

        return metrics

    def _step(self, batch, loss_name):
        img_input, img_tgt, i = batch

        img_input, img_tgt = img_input.unsqueeze(1), img_tgt.unsqueeze(1)

        img_pred = self.forward(img_input)

        loss = self._loss(img_pred, img_tgt)

        if loss_name != 'test':
            self.log(f'Loss/{loss_name}', loss.item(), prog_bar=True)

        return img_pred, loss

    def training_step(self, batch, batch_idx):
        # self.log('lr', self.lr_scheduler['scheduler'].get_last_lr()[0])
        _, loss = self._step(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        img_input, img_tgt, _ = batch
        img_output, loss = self._step(batch, 'val')

        dummy = -1* np.ones([256, 1])
        mat = np.concatenate(
            [img_input.cpu()[0], dummy, img_output.cpu()[0][0], dummy, img_tgt.cpu()[0]], axis=1)

        im = Image.fromarray(np.uint8(cm.gist_earth(mat)*255))
        images = wandb.Image(im, caption="Input-Output-Target")
        wandb.log({"Input-Output-Target": images})

        return loss

    def validation_step_end(self, batch) :
        pass

    def test_step(self, batch, batch_idx):
        img_input, img_tgt, i = batch
        img_pred, loss = self._step(batch, 'test')
        return img_pred, img_tgt, img_input, loss, i

    def test_step_end(self, output_results):
        img_pred , img_tgt, *_ = output_results
        loss = self._loss(img_pred , img_tgt)
        self.log("Loss/test", loss.item())
    
    def test_epoch_end(self, output_results):
        out_file = f"{self.hparams.logfile_name}/test_{self.hparams.test_prefix}.h5"
        print(f'Test results in:\t{out_file}')
        
        compression_lvl = 9
        sigmat_size = [256, 256]
        nimgs = len(self.trainer.datamodule.test_set)
        data = {}
        output_names = {'input', 'output', 'target'}
        with h5py.File(out_file, 'w', libver='latest') as h5_fh:
            for mode in output_names:
                data[mode] =\
                    h5_fh.create_dataset(mode,
                    shape=[nimgs] + sigmat_size, 
                    dtype=np.float32, chunks=tuple([1] + sigmat_size),
                    compression='gzip', compression_opts=compression_lvl)
        
        with h5py.File(out_file, 'a', libver='latest') as h5_fh:
            for batch in range(len(output_results)):
                img_output, img_tgt, img_input, loss, index = [x.cpu().numpy() for x in output_results[batch]]
                
                for i in range(img_output.shape[0]):
                    h5_fh['output'][index[i]] = img_output[i][0]
                    h5_fh['target'][index[i]] = img_tgt[i]
                    h5_fh['input'][index[i]] = img_input[i]

                dummy = -1* np.ones([256, 1])
                mat = np.concatenate(
                    [img_input[0], dummy, img_output[0][0], dummy, img_tgt[0]], axis=1)

                im = Image.fromarray(np.uint8(cm.gist_earth(mat)*255))
                images = wandb.Image(im, caption="Input-Output-Target")
                wandb.log({"TEST Input-Output-Target": images})
                
                f"{self.hparams.logfile_name}/test_{self.hparams.test_prefix}.h5"