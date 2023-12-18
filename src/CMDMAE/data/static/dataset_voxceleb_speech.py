from ...tools import load_audio_from_video, to_spec
from abc import ABC
from .base import BASE
from ..preprocess import Voxceleb
import numpy as np
import random
import torch
import librosa
from scipy.signal import lfilter


class SpeechVoxceleb_Static(BASE, ABC):
    def __init__(self, root: str, train: bool = True, spec_parameters: dict = None):
        super().__init__()
        self.vox = Voxceleb(root=root)
        self.vox.generate_table(number_id=None, number_part=None)
        self.table = self.vox.table
        self.train = train
        win_length = int(spec_parameters['win_length'] * 16000)
        self.spec_parameters = dict(n_fft=spec_parameters['n_fft'],
                                    hop=int(spec_parameters['hop']/2*win_length),
                                    # hop=int(spec_parameters['hop']*win_length),
                                    win_length=win_length)
        self.sr = 16000
        # -----
        self.index_wav = 0
        self.list_ = np.arange(0, len(self.table))
        self.number_frames = 0
        self.current_frame = 0
        self.shuffle = True
        random.shuffle(self.list_)

    def __len__(self):
        if self.train:
            return 76582387  # for three parts: parta, partb, partc
        else:
            return 7200945

    def H5_creation(self, path_to_save):
        pass

    @staticmethod
    def preemphasis(x, preemph):
        return lfilter([1, -preemph], [1], x)

    def load_wav(self, file: str, preemph: float = None):
        wav = load_audio_from_video(path_video=file, sr=self.sr)
        wav = librosa.to_mono(wav.transpose())
        wav = wav/np.max(np.abs(wav))
        if preemph is not None:
            wav = self.preemphasis(wav, preemph)
        return wav

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
                    wav = self.load_wav(file=file)
                    self.audio, _ = to_spec(wav, self.spec_parameters)
                    self.audio = (self.audio**2).transpose()
                except:
                    self.audio = torch.tensor([])
                self.index_wav += 1
                self.current_frame = 0
                self.number_frames = self.audio.shape[0]
                if self.number_frames > 0:
                    break
        self.i = self.audio[self.current_frame]
        self.current_frame += 1
        return torch.from_numpy(self.i).type(torch.FloatTensor).unsqueeze(0)
