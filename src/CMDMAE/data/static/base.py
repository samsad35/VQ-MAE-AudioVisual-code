from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BASE(ABC, Dataset):
    def __init__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def H5_creation(self, path_to_save: str):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass
