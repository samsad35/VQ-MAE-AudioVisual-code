from .ravdess import Ravdess
from .crema import Crema
from .enterface import Enterface
import torch
import numpy as np
from pathlib import Path
import librosa
from torch.utils.data import Dataset
import h5py


class EvaluationDataset(Dataset):
    def __init__(self,
                 root: str,
                 h5_path: str,
                 speaker_retain_test: list,
                 emotions_retain: list = None,
                 frames_per_clip: int = 50,
                 train: bool = True,
                 dataset: str = "ravdess"
                 ):
        super().__init__()
        if dataset.lower() == "ravdess":
            self.data = Ravdess(root=Path(root))
        elif dataset.lower() == "crema":
            self.data = Crema(root=Path(root))
        elif dataset.lower() == "enterface":
            self.data = Enterface(root=Path(root))
        self.data.generate_table()
        self.table = self.data.table
        self.emotions_retain = emotions_retain
        if speaker_retain_test is not None:
            self.speaker_retain_test = speaker_retain_test
            self.train = train
            self.table_()
        print(f"\t --> Evaluation of {dataset.upper()} | train: {train} | length: {len(self.table)}")
        # -----
        self.h5_path = h5_path
        self.h5_bool = h5_path is not None
        self.start_frame = 25
        self.seq_length = frames_per_clip
        win_length = int(64e-3 * 16000)
        hop = int(0.625 / 2 * win_length)
        self.spec_parameters = dict(n_fft=1024,
                                    hop=hop,
                                    win_length=win_length)

    def table_(self):
        if not self.train:
            self.table = self.table.loc[self.table['id'].isin(self.speaker_retain_test)].reset_index(drop=True)
            # self.table = self.table.loc[~self.table['emotion'].isin([-1])].reset_index(drop=True)
            if self.emotions_retain is not None:
                self.table = self.table.loc[self.table['emotion'].isin(self.emotions_retain)].reset_index(drop=True)

        else:
            self.table = self.table.loc[~self.table['id'].isin(self.speaker_retain_test)].reset_index(drop=True)
            self.table = self.table.loc[~self.table['name'].isin(["1032_ITS_SAD_XX.flv",
                                                                  "1032_IEO_DIS_MD.flv",
                                                                  "1032_IEO_HAP_MD.flv"])].reset_index(drop=True)
            self.table = self.table.loc[~self.table['emotion'].isin([-1])].reset_index(drop=True)
            if self.emotions_retain is not None:
                self.table = self.table.loc[self.table['emotion'].isin(self.emotions_retain)].reset_index(drop=True)

    def __len__(self):
        return len(self.table)

    def save_table(self, path: str):
        self.table.to_pickle(path)

    def get_weights(self, num_class: int):
        weights = []
        for i in range(num_class):
            w = len(self.table.loc[self.table['emotion'] == f"{i}"])
            w = 1 - (w / len(self.table))
            weights.append(w)
        return torch.tensor(weights)

    @staticmethod
    def load_wav(file: str):
        wav, sr = librosa.load(path=file, sr=16000)
        wav = librosa.to_mono(wav.transpose())
        wav = wav / np.max(np.abs(wav))
        return wav

    def get_information(self, index):
        path = self.table.iloc[index]['path']
        id = self.table.iloc[index]['id']
        emotion = self.table.iloc[index]['emotion']
        name = self.table.iloc[index]['name']
        return emotion, id, path, name

    @staticmethod
    def padding(data, seq_length=50):
        """

        :param seq_length:
        :param data:
        :return:
        """
        if len(data.shape) == 2:
            data = np.pad(data, ((0, seq_length - data.shape[0]), (0, 0)), 'wrap')
        return data

    def open(self):
        self.hdf5 = h5py.File(self.h5_path, mode='r')

    def read(self, id, name):
        a = np.array(self.hdf5[f'/{id}/audio_{name}'])
        v = np.array(self.hdf5[f'/{id}/visual_{name}'])
        return a, v

    def __getitem__(self, item):
        if not hasattr(self, 'hdf5'):
            self.open()
        emotion, id, path, name = self.get_information(item)
        a, v = self.read(id, name)
        if a.shape[0] < v.shape[0] * 2:
            a = self.padding(a, seq_length=v.shape[0] * 2)
        else:
            v = self.padding(v, seq_length=a.shape[0] // 2)
        number_frames = v.shape[0]
        if number_frames < self.seq_length + 30:
            a = self.padding(a, seq_length=(self.seq_length + 30) * 2)
            v = self.padding(v, seq_length=self.seq_length + 30)
        a = torch.from_numpy(a)
        v = torch.from_numpy(v)
        # current_frame = np.random.randint(0, number_frames - self.seq_length)
        current_frame = 25
        self.i_1 = a[(current_frame * 2):(current_frame * 2) + (self.seq_length * 2)]
        self.i_2 = v[current_frame:current_frame + self.seq_length]
        return self.i_1.type(torch.LongTensor), self.i_2.type(torch.LongTensor), emotion
