from abc import ABC
from .base import BASE
from ...tools import read_video_decord
from ..preprocess import Voxceleb
import torch
import random
import torchvision.transforms as transforms
import numpy as np


class VoxcelebSequential(BASE, ABC):
    def __init__(self,
                 root: str,
                 frames_per_clip: int = 50,
                 hop_length: float = 100,
                 transform: transforms = None,
                 train: bool = True):
        super().__init__()
        self.vox = Voxceleb(root=root)
        if train:
            self.vox.generate_table(number_id=1)
        else:
            self.vox.generate_table(number_id=1)
        self.table = self.vox.table
        self.transform = transform
        # -----
        self.train = train
        self.index_wav = 0
        self.list_ = np.arange(0, len(self.table))
        self.number_frames = 0
        self.current_frame = 0
        self.shuffle = True
        self.seq_length = frames_per_clip
        self.hop_length = 50
        self.hop_length = hop_length
        random.shuffle(self.list_)
        self.__len__()

    def __len__(self):
        # length = 0
        # for _, _, _, file in self.vox.generator(number_id=1):
        #     frames = read_video_decord(file)
        #     current_frame = 0
        #     while (frames.shape[0] - current_frame) > self.seq_length:
        #         length += 1
        #         current_frame += self.seq_length
        #
        #     print(length)
        if self.train:
            return 544
        else:
            return 544

    def H5_creation(self, path_to_save):
        pass

    def reset(self):
        if self.index_wav >= len(self.list_):
            self.index_wav = 0
            if self.shuffle:
                random.shuffle(self.list_)

    def __getitem__(self, item):
        if (self.number_frames - self.current_frame) < self.seq_length:
            while True:
                self.reset()
                file = self.table.iloc[self.list_[self.index_wav]]['file_path']
                self.visual = read_video_decord(file_path=file)
                self.visual = self.visual
                self.index_wav += 1
                self.current_frame = 0
                self.number_frames = self.visual.shape[0]
                if self.number_frames >= self.seq_length:
                    break
        self.i = self.visual[self.current_frame: self.current_frame + self.seq_length].transpose((0, 3, 1, 2))
        self.current_frame += int(self.hop_length / 100 * self.seq_length)
        self.i = torch.from_numpy(self.i)/255.0
        if self.transform is not None:
            self.i = self.transform(self.i)
        return self.i.type(torch.FloatTensor)
