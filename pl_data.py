from dataLoader import UnpairedDataset, UnpairedDatasetImages
import torch
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl

class OADataModule(pl.LightningDataModule):
    def __init__(self, opt):
        """
        :param directory: directory with "TYPE" placeholder being replaced with "train", "val", "test"
        """
        super().__init__()

        if opt.test:
            self.idx_test = []
        else:
            self.idx_test = [0, 142, 285, 428, 570, 713, 856, 998, 1141, 1284, 1426,
                1569, 1712, 1854, 1997, 2140, 2282, 2425, 2568, 2710, 2853, 2996, 3138, 
                3281, 3424, 3566, 3709, 3851, 3994, 4137, 4280, 4422]

        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.test_labels = None
        self.weights = None

        self.opt = opt

        self.batch_size = self.opt.batch_size
        # self.idx_to_token = idx_to_token

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if self.opt.mode in ['styleImages']:
            dataset = UnpairedDatasetImages(
                self.opt.file_in,
                file_name_real=self.opt.target)
        else:
            dataset = UnpairedDataset(
                self.opt.file_in,
                file_name_real=self.opt.target,
                geometry=self.opt.geometry
                )

        idx_all = list(set(range(len(dataset))) - set(self.idx_test))

        # separate the leftover of the dataset into train and validation
        idx_train, idx_val = train_test_split(
            idx_all, test_size=32, random_state=42)

        if stage in (None, "fit"):
            self.val_set = torch.utils.data.dataset.Subset(
                dataset, torch.LongTensor(idx_val))
            self.train_set = torch.utils.data.dataset.Subset(
                dataset, torch.LongTensor(idx_train))

        print("Number of train points:", len(self.train_set))

        if stage in (None, "test"):
            self.test_set = torch.utils.data.dataset.Subset(
                dataset, torch.LongTensor(self.idx_test))
            print("Number of test points:", len(self.test_set))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.opt.num_workers,
            prefetch_factor=1,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.opt.num_workers,
        )