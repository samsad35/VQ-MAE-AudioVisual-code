from abc import ABC
from .base import BASE
from ..preprocess import Voxceleb
import torch
import random
import torchvision.transforms as transforms
import numpy as np


class VoxcelebSequentialMultimodalPT(BASE, ABC):
    def __init__(self,
                 frames_per_clip: int = 50,
                 hop_length: float = 100,
                 transform: transforms = None,
                 train: bool = True,
                 root_modality_1: str = None,
                 root_modality_2: str = None):
        super().__init__()
        self.vox = Voxceleb(root=root_modality_1, ext="pt")
        if train:
            self.vox.generate_table(number_id=None)
        else:
            self.vox.generate_table(number_id=None)
        self.table = self.vox.table
        self.transform = transform
        self.root_modality_1 = root_modality_1
        self.root_modality_2 = root_modality_2
        self.h5_bool = True
        # -----
        self.train = train
        self.list_ = list(np.arange(0, len(self.table)))
        self.number_frames = 0
        self.current_frame = 0
        self.shuffle = True
        self.seq_length = frames_per_clip
        self.hop_length = hop_length
        random.shuffle(self.list_)

    def __len__(self):
        if self.train:
            return len(self.table)  # 126896  # 1712
        else:
            return int(126896/100)

    def open(self):
        pass

    def read(self, indx: tuple):
        return torch.load(f'{self.root_modality_1}/{indx[0]}/{indx[1]}/{indx[2]}/{indx[3]}'), \
               torch.load(f'{self.root_modality_1}/{indx[0]}/{indx[1]}/{indx[2]}/{indx[3]}'),

    def get_information(self, index):
        part = self.table.iloc[index]['part']
        id = self.table.iloc[index]['id']
        ytb_id = self.table.iloc[index]['ytb_id']
        name = self.table.iloc[index]['name']
        return part, id, ytb_id, name

    def __getitem__(self, item):
        while True:
            info = self.get_information(self.list_[item])
            self.modality_1, self.modality_2 = self.read(info)
            self.modality_1, self.modality_2 = self.modality_1,  self.modality_2
            self.number_frames = self.modality_1.shape[0]
            if self.number_frames >= self.seq_length:
                break
            else:
                item += 1
        self.current_frame = np.random.randint(0, self.number_frames - self.seq_length)
        self.i_1 = self.modality_1[self.current_frame: self.current_frame + self.seq_length]
        self.i_2 = self.modality_2[self.current_frame: self.current_frame + self.seq_length]

        return self.i_1.type(torch.LongTensor), self.i_2.type(torch.LongTensor)
