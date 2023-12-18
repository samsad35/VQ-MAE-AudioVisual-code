import numpy as np
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from ...model import VQVAE
import h5py
import os
from tqdm import tqdm
from einops import repeat, rearrange


class StlDatasetStatic(Dataset):
    def __init__(self, h5_path: str = None):
        self.h5_path = h5_path
        if self.h5_path is not None:
            self.data = h5py.File(r'H5/STL.hdf5', 'r')
        else:
            self.data = datasets.STL10(root="data/STL10", split="train+unlabeled", download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
                                       ]))
        self.length = self.__len__()

    def __len__(self):
        return 105000

    def h5_creation(self, h5_path: str):
        # """ VQVAE """
        num_hiddens = 128
        num_residual_hiddens = 64
        num_residual_layers = 2
        embedding_dim = 64
        num_embeddings = 512
        commitment_cost = 0.25
        decay = 0.99
        vqvae = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
                      num_embeddings, embedding_dim,
                      commitment_cost, decay)
        vqvae.load(r"checkpoint/STL10/model_checkpoint_Y2022M11D7")
        vqvae.to('cuda')
        vqvae.eval()
        if os.path.isfile(h5_path):
            os.remove(h5_path)
        file_h5 = h5py.File(h5_path, 'a')
        print("Create a new H5")
        for index in tqdm(range(self.length), desc='H5-STL10'):
            image = self.data[index][0]
            image = rearrange(image, 'c (h s1) (w s2) -> (h w) c s1 s2', c=3, s1=16, s2=16).to("cuda")
            image = vqvae.pre_vq_conv(vqvae.encoder(image))
            indices, input_shape, device = vqvae.vq_vae.get_code_indices(image)
            indices = torch.reshape(indices, (-1, 16))
            file_h5.create_dataset(name=str(index), data=indices.cpu().detach().numpy())
        file_h5.flush()
        file_h5.close()


    def __getitem__(self, item):
        if self.h5_path is not None:
            return torch.from_numpy(np.array(self.data[f'{item}'])).type(torch.long)
        else:
            return self.data[item]
