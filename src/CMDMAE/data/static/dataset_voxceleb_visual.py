from abc import ABC
from .base import BASE
from ...tools import read_video_decord
from ..preprocess import Voxceleb
import torch
import random
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class Voxceleb_Static(BASE, ABC):
    def __init__(self, root: str, train: bool = True, transform: transforms = None):
        super().__init__()
        self.vox = Voxceleb(root=root)
        self.vox.generate_table(number_id=None, number_part=None)
        self.table = self.vox.table
        self.transform = transform
        self.train = train
        # -----
        self.index_wav = 0
        self.list_ = np.arange(0, len(self.table))
        self.number_frames = 0
        self.current_frame = 0
        self.shuffle = True
        random.shuffle(self.list_)

    def __len__(self):
        # length = 0
        # with tqdm(total=len(self.table)) as pbar:
        #     for _, _, _, _, file in self.vox.generator(number_id=None):
        #         try:
        #             frames = read_video_decord(file)
        #             length += frames.shape[0]
        #         except:
        #             print(file)
        #         pbar.update(1)
        #     print(length)
        if self.train:
            return 76582387  # for three parts: parta, partb, partc
        else:
            return 7200945

    def H5_creation(self, path_to_save):
        pass

    def reset(self):
        if self.index_wav >= len(self.list_):
            self.index_wav = 0
            if self.shuffle:
                random.shuffle(self.list_)

    def __getitem__(self, item):
        if self.current_frame == self.number_frames:
            while True:
                self.reset()
                file = self.table.iloc[self.list_[self.index_wav]]['file_path']
                try:
                    self.visual = read_video_decord(file_path=file)
                except:
                    self.visual = torch.tensor([])
                self.index_wav += 1
                self.current_frame = 0
                self.number_frames = self.visual.shape[0]
                if self.number_frames > 0:
                    break
        self.i = self.visual[self.current_frame]
        if self.transform is not None:
            self.i = self.transform(self.i)
        self.current_frame += 1
        return self.i.type(torch.FloatTensor)
