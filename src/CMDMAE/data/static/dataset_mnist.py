import numpy as np
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torchvision.transforms as transforms


class MnistDatasetStatic(Dataset):
    def __init__(self, directory_name: str = None,
                 section: str = None,
                 h5_path: str = None,
                 transform: transforms = None):
        self.section = section
        self.h5_path = h5_path
        self.data = np.load(directory_name + r"\mnist_test_seq.npy")
        self.data = self.data.transpose((1, 0, 2, 3))[:, :, None, ...]
        self.data = self.data.reshape((10000 * 20, 1, 64, 64))
        self.length = self.data.shape[0]
        if self.section == "train":
            self.data = self.data[:int(self.length * 0.9), ...]
        else:
            self.data = self.data[int(self.length * 0.9):, ...]
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        if self.transform is not None:
            image = self.transform(self.data[item].transpose(1, 2, 0))
        else:
            image = torch.from_numpy(self.data[item])
        return image.type(torch.FloatTensor)

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
    data = MnistDatasetStatic(directory_name=r"D:\These\data\Audio-Visual\moving-MNIST",
                              section='train')
    a = data[20000]
    print(a.shape)
    a = np.repeat(a, 20, axis=0)
    print(a.shape)
    data.plot(a)
