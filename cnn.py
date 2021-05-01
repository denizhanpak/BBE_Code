import os
from typing import Any, Callable, List, Optional, Tuple, Union

from torchvision.models import resnet18

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from pl_bolts.models.autoencoders import VAE

from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F


class FishDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        labels_file: str,
        transform: Optional[Callable] = T.ToTensor()
    ):
        self.transform = transform
        self.dataframe = self.create_dataframe(data_dir, labels_file)

    def create_dataframe(self, data_dir: str, labels_file: str) -> pd.DataFrame:
        #Get Labels
        labels = pd.read_csv(labels_file)
        images = [f.path for f in os.scandir(data_dir) if f.name.endswith(".png")]
        
        tmp = dict()
        for image in images:
            index = image[-9:-4]
            index = index.strip("_")
            index = int(index) - 1
            tmp[index] = image

        l = []
        for i in range(min(len(tmp), len(labels))):
            l.append(tmp[i])

        labels["image_path"] = l

        return labels


    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        image_path = self.dataframe["image_path"][idx]
        image = Image.open(image_path).convert("RGB")
        label = []
        label.append(self.dataframe["0"][idx])
        label.append(self.dataframe["1"][idx])
        label.append(self.dataframe["2"][idx])
        label.append(self.dataframe["3"][idx])
        label.append(self.dataframe["4"][idx])

        if self.transform is not None:
            image = self.transform(image)

        image = image[:,:,1:]
        return image, torch.tensor(label, dtype=torch.float)

#print(FishDataset(data_dir="/home/denpak/Research/BBEProject/Recordings",labels_file="labels.csv")[0][0].shape)



class FishDataModule(LightningDataModule):
    name = "fish_encoder"
    dataset_cls = FishDataset
    dims = (3, 64, 64)

    def __init__(
        self,
        val_fold: int = 0,
        num_workers: int = 16,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            val_fold: Index of validation fold (in range [0, 5])
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__()
        self.val_fold = val_fold
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    @property
    def default_transforms(self) -> Callable:
        """ Return default data transformation. """
        return T.ToTensor()

    def setup(self, stage: Optional[str] = None) -> None:
        """ Create train, val, and test dataset. """
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms if self.val_transforms is None else self.val_transforms
            self.dataset = self.dataset_cls(data_dir="/home/denpak/Research/BBEProject/Recordings",labels_file="labels.csv",transform=train_transforms)

            val_size = int(0.3 * len(self.dataset))
            train_size = len(self.dataset) - val_size
            self.dataset_train, self.dataset_val = random_split(self.dataset , [val_size, train_size])

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms if self.test_transforms is None else self.test_transforms
            self.dataset_test = self.dataset_cls(transform=test_transforms)

    def train_dataloader(self) -> DataLoader:
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        return self._data_loader(self.dataset_val)

    def test_dataloader(self) -> DataLoader:
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )


class FishClassifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.mse_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.mse_loss(logits, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


#model = VAE(64,latent_dim=5)
#trainer = pl.Trainer(gpus=1)
#data = FishDataModule()
#v = VAE.load_from_checkpoint(")
PATH = "/home/denpak/Research/BBEEncoders/cnn_model"
ckpt = torch.load(PATH+"/last.ckpt")
model = resnet18(num_classes=5)
keys = ckpt['state_dict'].keys()
state_dict = {}
for key in keys:
    state_dict[key[6:]] = ckpt['state_dict'][key]
print(state_dict.keys())
model.load_state_dict(state_dict)
#model = FishClassifier(model)
#model.load_from_checkpoint(ckpt, map_location = {'cuda:1':'cpu'})
d = FishDataset(data_dir="/home/denpak/Research/BBEProject/Recordings",labels_file="labels.csv")[0][0]
x = torch.unsqueeze(d,0)
print(model(x))
