import torch.utils.data as data
from abc import ABC, abstractmethod

class BaseDataset(data.Dataset, ABC):
    """Abstract Base Class for all Datasets"""
    def __init__(self, args):
        self.args = args
        self.root = args.data_dir
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass