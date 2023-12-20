import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from six.moves import xrange
import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# from dalle_pytorch import DiscreteVAE
from .follow_up_vqvae import Follow



class Train_DiscreteVAE:
    def __init__(self, training_data: Dataset, validation_data: Dataset,
                 config_training: dict = None):
        self.config_training = config_training
        self.model = DiscreteVAE(
            image_size=128,
            num_layers=2,  # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
            num_tokens=512,
            # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
            codebook_dim=8,  # codebook dimension
            hidden_dim=64,  # hidden dimension
            num_resnet_blocks=1,  # number of resnet blocks
            temperature=0.9,  # gumbel softmax temperature, the lower this is, the harder the discretization
            straight_through=False,  # straight-through for gumbel softmax. unclear if it is better one way or the other
            smooth_l1_loss=True
        )
        self.load_epoch = 0
        self.parameters = dict()
        self.training_data = training_data
        self.training_loader = DataLoader(training_data, batch_size=config_training['batch_size'], shuffle=True,
                                          pin_memory=True)
        self.validation_loader = DataLoader(validation_data, batch_size=16, shuffle=False, pin_memory=True)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config_training['lr'], amsgrad=False)

        self.device = torch.device(self.config_training['device'])
        self.model = self.model.to(device=self.device)

        self.follow = Follow("vqvae", dir_save=r"checkpoint/VQVAE", variable=vars(self.model))
        self.train_res_recon_error = []

        """ TensorBoard """
        # self.writer = SummaryWriter(os.path.join('logs', 'mnist-moving', self.to_day))

    def save(self, parameters: dict):
        torch.save(parameters, f'checkpoint\\model_checkpoint_{self.to_day}')

    def load(self, path: str = "", optimizer: bool = True):
        print("LOAD [", end="")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        if optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.load_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"model:ok  | optimizer:{optimizer}  |  loss: {loss}  |  epoch: {self.load_epoch}]")

    def train_process(self):
        self.model.train()
        num_training_updates = 15000 * 20
        with tqdm(total=num_training_updates - self.load_epoch, desc=f"Training ") as pbar:
            for i in xrange(num_training_updates - self.load_epoch):
                (data) = next(iter(self.training_loader))
                data = data.to(self.device)
                self.optimizer.zero_grad()
                # loss, data_recon = self.model(data, return_loss=True, return_recons=True)  # Discrete VAE DALLE
                loss, data_recon = self.model(data, return_loss=True, return_recons=True)
                loss.backward()
                self.train_res_recon_error.append(loss.item())
                self.optimizer.step()
                if (i + 1) % 500 == 0:
                    self.plot_images(data_recon[:16], show=False, save="temps-reconstruction.png")
                    self.plot_images(data[:16], show=False, save="temps-real.png")
                    self.parameters = dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                                           epoch=i, loss=loss)
                    self.follow(epoch=i, loss_train=loss.item(), parameters=self.parameters)
                    pbar.set_description(f"Training: recon-loss: {loss:.4f}")
                    self.plot()
                pbar.update(1)

    def plot(self):
        train_res_recon_error_smooth = savgol_filter(self.train_res_recon_error, 201, 7)
        f = plt.figure(figsize=(16, 8))
        ax = f.add_subplot(1, 2, 1)
        ax.plot(train_res_recon_error_smooth)
        ax.set_yscale('log')
        ax.set_title('Smoothed NMSE.')
        ax.set_xlabel('iteration')
        plt.savefig(r'plot.jpeg')
        plt.close()

    def __call__(self, *args, **kwargs):
        self.train_process()

    @staticmethod
    def plot_images(images, show: bool = True, save: str = None):
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
                ax.imshow(images[i].permute(1, 2, 0).cpu().detach().numpy() + 0.5)
                plt.axis('off')
                i = i + 1
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)
            plt.close()

    def reconstruction_vqvae(self, indices):
        valid_quantize = self.model.vq_vae.vq(indices.type(torch.int64), torch.Size([20, 16, 16, 64]), "cuda")
        return self.model.decoder(valid_quantize)

    def view_reconstruction(self, rep=1):
        self.model.eval()
        # vqgan = VQGanVAE()
        # vqgan.to('cuda')
        for valid_originals in self.validation_loader:
            indices = self.model.get_codebook_indices(valid_originals.to("cuda"))
            # indices = vqgan.get_codebook_indices(valid_originals.to("cuda"))
            images = self.model.decode(indices)
            # images = vqgan.decode(indices)
            self.plot_images(images, show=True)
            self.plot_images(valid_originals, show=True)




