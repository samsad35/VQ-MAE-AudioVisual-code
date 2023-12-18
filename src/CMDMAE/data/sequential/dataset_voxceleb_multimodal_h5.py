from abc import ABC
from .base import BASE
from ..preprocess import Voxceleb
import torch
import random
import torchvision.transforms as transforms
import numpy as np
import h5py
import pandas


class VoxcelebSequentialMultimodalH5(BASE, ABC):
    def __init__(self,
                 root: str,
                 frames_per_clip: int = 50,
                 hop_length: float = 100,
                 transform: transforms = None,
                 train: bool = True,
                 table_path: str = None,
                 h5_path_1: str = None,
                 h5_path_2: str = None):
        super().__init__()
        if table_path is None:
            self.vox = Voxceleb(root=root)
            self.vox.generate_table(number_id=None)
            self.table = self.vox.table
            # print(self.table.info())
        else:
            self.table = pandas.read_pickle(table_path)
        self.transform = transform
        self.h5_path_1 = h5_path_1
        self.h5_path_2 = h5_path_2
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
        return len(self.table)  # 126896  # 1712

    def save_table(self, path: str):
        self.table.to_pickle(path)

    def open(self):
        self.m1_hdf5 = h5py.File(self.h5_path_1, mode='r')
        self.m2_hdf5 = h5py.File(self.h5_path_2, mode='r')

    def read(self, indx: tuple):
        m1 = np.array(self.m1_hdf5[f'/{indx[0]}/{indx[1]}/{indx[2]}/{indx[3]}'])
        m2 = np.array(self.m2_hdf5[f'/{indx[0]}/{indx[1]}/{indx[2]}/{indx[3]}'])
        return m1, m2

    def get_information(self, index):
        part = self.table.iloc[index]['part']
        id = self.table.iloc[index]['id']
        ytb_id = self.table.iloc[index]['ytb_id']
        name = self.table.iloc[index]['name']
        return part, id, ytb_id, name

    def __getitem__(self, item):
        if not hasattr(self, 'm1_hdf5'):
            self.open()
        while True:
            info = self.get_information(self.list_[item])
            try:
                self.modality_1, self.modality_2 = self.read(info)
            except:
                item += 1
                continue
            self.modality_1, self.modality_2 = torch.from_numpy(self.modality_1),  torch.from_numpy(self.modality_2)
            self.number_frames = self.modality_1.shape[0]
            if self.number_frames >= self.seq_length:
                break
            else:
                item += 1
        self.current_frame = np.random.randint(0, self.number_frames - self.seq_length)
        self.i_1 = self.modality_1[self.current_frame: self.current_frame + self.seq_length]
        # self.i_2 = self.modality_2[self.current_frame: self.current_frame + self.seq_length]
        self.i_2 = self.modality_2[(self.current_frame*2): (self.current_frame*2) + (self.seq_length*2)]

        return self.i_1.type(torch.LongTensor), self.i_2.type(torch.LongTensor)
