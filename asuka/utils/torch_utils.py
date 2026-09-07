from abc import abstractmethod

import torch
from torch.utils.data import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AbstractTorchDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.imgs = list()

    @abstractmethod
    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.imgs)
