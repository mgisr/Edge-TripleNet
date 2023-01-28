from typing import Tuple

import h5py
from numpy import array
from torch.utils.data import (Dataset, DataLoader, random_split)
import pytorch_lightning as pl


class EdgeTrainDataset(Dataset):
    def __init__(self, transform=None) -> None:
        super().__init__()
        with h5py.File('data/mnist.h5', 'r') as f:
            self.train_data = array(f['train_data'])
            self.train_edge_data = array(f['train_edge_data'])
            self.train_label = array(f['train_label'])
        self.transform = transform

    def __getitem__(self, index) -> Tuple:
        X, Y, z = self.train_data[index], self.train_edge_data[index], self.train_label[index]
        if self.transform:
            X = self.transform(X)
            Y = self.transform(Y)
        return X, Y, z

    def __len__(self) -> int:
        return len(self.train_label)


class EdgeTestDataset(Dataset):
    def __init__(self, transform=None) -> None:
        super().__init__()
        with h5py.File('data/mnist.h5', 'r') as f:
            self.test_data = array(f['test_data'])
            self.test_edge_data = array(f['test_edge_data'])
            self.test_label = array(f['test_label'])
        self.transform = transform

    def __getitem__(self, index) -> Tuple:
        X, Y, z = self.test_data[index], self.test_edge_data[index], self.test_label[index]
        if self.transform:
            X = self.transform(X)
            Y = self.transform(Y)
            z = self.transform(z)
        return X, Y, z

    def __len__(self) -> int:
        return len(self.test_label)


class LightEdgeDataset(pl.LightningDataModule):
    def __init__(self, batch_size=32, transform=None) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage: str = None) -> None:
        if stage == 'fit' or stage is None:
            full_dataset = EdgeTrainDataset(self.transform)
            self.train_dataset, self.val_dataset = random_split(full_dataset, [55000, 5000])
        if stage == 'test' or stage is None:
            self.test_dataset = EdgeTestDataset(self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, True, num_workers=8)
