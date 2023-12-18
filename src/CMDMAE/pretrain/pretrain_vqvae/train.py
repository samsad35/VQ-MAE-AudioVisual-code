from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from ...base import Train
from six.moves import xrange
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from .follow_up_vqvae import Follow
from scipy.signal import savgol_filter
# from dalle_pytorch import DiscreteVAE
from ...model import VQVAE
import torch.nn.functional as F
from torchvision.utils import make_grid
import kornia as K


class VQVAE_Train(Train):
    def __init__(self, model: VQVAE, training_data: Dataset, validation_data: Dataset,
                 config_training: dict = None):

        """ Model """
        self.model = model

        """ Dataloader """
        self.training_loader = DataLoader(training_data, batch_size=config_training['batch_size'], shuffle=True,
                                          pin_memory=True, num_workers=config_training['num_workers'])
        self.validation_loader = DataLoader(validation_data, batch_size=16, shuffle=False, pin_memory=True)

        """ Optimizer """
        self.device = torch.device(config_training['device'])
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config_training['lr'], amsgrad=False)

        """ Config """
        self.config_training = config_training
        self.train_res_recon_error = []
        self.train_res_perplexity = []
        self.load_epoch = 0
        self.parameters = dict()

        """ Follow """
        self.follow = Follow("vqvae", dir_save=r"checkpoint", variable=vars(self.model))
        self.epochs = config_training['epochs']

        """ Loss function """
        self.loss_fn = F.smooth_l1_loss if config_training['smooth_l1_loss'] else F.mse_loss

    def one_epoch(self):
        pass

    def fit(self):
        self.model.train()
        num_training_updates = len(self.training_loader) * self.epochs
        i = self.load_epoch
        while i < num_training_updates:
            for data in tqdm(self.training_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                vq_loss, data_recon, perplexity = self.model(data)
                recon_error = self.loss_fn(data, data_recon)
                loss = recon_error + vq_loss
                loss.backward()
                self.train_res_recon_error.append(loss.item())
                self.train_res_perplexity.append(perplexity.item())
                self.optimizer.step()
                i += 1
                if (i + 1) % 500 == 0:
                    self.plot_images(data_recon[:16], show=False,
                                     save=f"{self.follow.path_samples}/temps-reconstruction.png")
                    self.plot_images(data[:16], show=False, save=f"{self.follow.path_samples}/temps-real.png")
                    self.parameters = dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                                           epoch=i, loss=loss)
                    self.follow(epoch=i, loss_train=loss.item(), loss_validation=loss.item(),
                                parameters=self.parameters)
                    print(
                        f'In iter. {i}, average traning loss is {loss.item():.4f}'
                        f' and average perplexity is {perplexity.item():.4f}')
                    self.plot()

    def plot(self):
        train_res_recon_error_smooth = savgol_filter(self.train_res_recon_error, 201, 7)
        train_res_perplexity_smooth = savgol_filter(self.train_res_perplexity, 201, 7)
        f = plt.figure(figsize=(16, 8))
        ax = f.add_subplot(1, 2, 1)
        ax.plot(train_res_recon_error_smooth)
        ax.set_yscale('log')
        ax.set_title('Smoothed NMSE.')
        ax.set_xlabel('iteration')
        ax = f.add_subplot(1, 2, 2)
        ax.plot(train_res_perplexity_smooth)
        ax.set_title('Smoothed Average codebook usage (perplexity).')
        ax.set_xlabel('iteration')
        plt.savefig(f"{self.follow.path}/loss_perplexity.png")
        plt.close()

    @staticmethod
    def plot_images_(images, show: bool = True, save: str = None):
        fig = plt.figure(figsize=(10, 10))
        if len(images) > 8:
            nrows = 4
            ncols = 4
        else:
            ncols = len(images)
            nrows = 1

        gs = GridSpec(ncols=ncols, nrows=nrows)
        i = 0
        for line in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[line, col])
                if images[i].shape[0] == 1:
                    ax.imshow(images[i][0, :, :].cpu().detach().numpy() + 0.5)
                else:
                    ax.imshow(images[i].permute(1, 2, 0).cpu().detach().numpy() + 0.5)
                plt.axis('off')
                i = i + 1
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)
            plt.close()

    @staticmethod
    def plot_images(images, show: bool = True, save: str = None):
        plt.figure(figsize=(10, 10))
        out: torch.Tensor = make_grid(images + 0.5, nrow=4, padding=5)
        out_np: np.array = K.tensor_to_image(out)
        plt.imshow(out_np)
        plt.axis('off')
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)
            plt.close()

    def eval(self):
        pass

    def load(self, path: str = "", optimizer: bool = True):
        print("LOAD [", end="")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        if optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.load_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"model: ok  | optimizer:{optimizer}  |  loss: {loss}  |  epoch: {self.load_epoch}]")
