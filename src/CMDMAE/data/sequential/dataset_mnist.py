import numpy as np
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torchvision.transforms as transforms
# from ...model import VQVAE
import pickle


class MnistDatasetSequential(Dataset):
    def __init__(self, root: str, section: str = None):
        self.section = section
        self.data = np.load(root)
        self.data = self.data.transpose((1, 0, 2, 3))[:, :, None, ...]
        self.length = self.data.shape[0]
        if self.section.lower() == "train".lower():
            self.data = self.data[:int(self.length * 0.9), ...]
        else:
            self.data = self.data[int(self.length * 0.9):, ...]


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return (torch.from_numpy(self.data[item]).type(torch.FloatTensor)/255.0)-0.5
        # return (torch.from_numpy(self.data[item]).type(torch.FloatTensor)/255.0)

    @staticmethod
    def plot(images, show: bool = True, save: str = None):
        fig = plt.figure(figsize=(8, 4))
        if len(images) > 10:
            nrows = int(len(images) / 10)
            ncols = 10
        else:
            ncols = len(images)
            nrows = 1

        gs = GridSpec(ncols=ncols, nrows=nrows)
        i = 0
        for line in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[line, col])
                ax.imshow(images[i].cpu().detach().numpy(), 'gray')
                plt.axis('off')
                i = i + 1
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)
            plt.close()


if __name__ == '__main__':
    data = MnistDatasetSequential(section='train')
    a = data[20]
    print(a.shape)
    data.plot(a[:, 0])
