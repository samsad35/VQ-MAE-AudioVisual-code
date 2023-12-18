from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import numpy as np
from ...base import Train
from ...model import VQVAE, MAE
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from .follow_up_mae import Follow
import math
from einops import repeat, rearrange
from math import log2, sqrt
torch.cuda.empty_cache()


class MAE_Train(Train):
    def __init__(self, mae: MAE, vqvae: VQVAE, training_data: Dataset, validation_data: Dataset,
                 config_training: dict = None):
        self.device = torch.device(config_training['device'])
        """ Model """
        self.model = mae
        self.vqvae = vqvae
        self.model.to(self.device)
        self.vqvae.to(self.device)

        """ Dataloader """
        self.training_loader = DataLoader(training_data, batch_size=config_training['batch_size'], shuffle=True,
                                          pin_memory=True)
        self.validation_loader = DataLoader(validation_data, batch_size=config_training['batch_size'], shuffle=True,
                                            pin_memory=True)

        """ Optimizer """
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=config_training['lr'] * config_training['batch_size'] / 256,
                                           betas=(0.9, 0.95),
                                           weight_decay=config_training["weight_decay"])
        lr_func = lambda epoch: min((epoch + 1) / (config_training["warmup_epoch"] + 1e-8),
                                    0.5 * (math.cos(epoch / config_training["total_epoch"] * math.pi) + 1))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_func, verbose=True)

        """ Loss """
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")

        """ Config """
        self.config_training = config_training
        self.load_epoch = 0
        self.step_count = 0
        self.parameters = dict()

        """ Follow """
        self.follow = Follow("mae", dir_save=r"checkpoint", variable=vars(self.model))

    def one_epoch(self):
        self.model.train()
        losses = []
        for img in tqdm(iter(self.training_loader)):
            self.optimizer.zero_grad()
            self.step_count += 1
            img = img.to(self.device)
            batch = img.shape[0]
            img = rearrange(img, 'b t c h w -> (b t) c h w')
            indices = self.vqvae.get_codebook_indices(img)
            indices = torch.reshape(indices, (batch, -1, 16 * 16))
            predicted_indices, mask = self.model(indices)
            loss = self.criterion(predicted_indices.flatten(0, 2)[mask.flatten(0).to(torch.bool)],
                                  indices.flatten(0)[mask.flatten(0).to(torch.bool)].to(torch.long))
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return losses

    def fit(self):
        for e in range(self.config_training["total_epoch"]):
            losses = self.one_epoch()
            losses_val = self.eval()
            self.lr_scheduler.step()
            avg_loss_train = sum(losses) / len(losses)
            avg_loss_val = sum(losses_val) / len(losses_val)
            self.parameters = dict(model=self.model.state_dict(),
                                   optimizer=self.optimizer.state_dict(),
                                   scheduler=self.lr_scheduler.state_dict(),
                                   epoch=e,
                                   loss=avg_loss_train)
            print(
                f'In epoch {e}, average traning loss is {avg_loss_train}. and average validation loss is {avg_loss_val}')
            self.follow(epoch=e, loss_train=avg_loss_train, loss_validation=avg_loss_val, parameters=self.parameters)

    def plot_train(self):
        pass

    def plot_images(self, indices, show: bool = True, save: str = None):
        images = self.vqvae.decode(indices[0])

        def plot(images, show: bool = True, save: str = None):
            fig = plt.figure(figsize=(8, 4))
            if len(images) > 10:
                nrows = 5
                ncols = 10
            else:
                ncols = len(images)
                nrows = 1

            gs = GridSpec(ncols=ncols, nrows=nrows)
            i = 0
            for line in range(nrows):
                for col in range(ncols):
                    ax = fig.add_subplot(gs[line, col])
                    if self.vqvae.channels == 1:
                        ax.imshow(images[i].cpu().detach().numpy() + 0.5, 'gray')
                    else:
                        ax.imshow(images[i].cpu().detach().numpy().transpose(1, 2, 0) + 0.5)
                    plt.axis('off')
                    i = i + 1
            if show:
                plt.show()
            if save is not None:
                plt.savefig(save)
                plt.close()

        if self.vqvae.channels == 1:
            plot(images[:, 0], show=show, save=save)
        else:
            plot(images[:], show=show, save=save)

    def eval(self):
        self.model.eval()
        losses = []
        for img in tqdm(iter(self.validation_loader)):
            img = img.to(self.device)
            batch = img.shape[0]
            img = rearrange(img, 'b t c h w -> (b t) c h w')
            indices = self.vqvae.get_codebook_indices(img)
            indices = torch.reshape(indices, (batch, -1, 16 * 16))
            predicted_indices, mask = self.model(indices)
            loss = self.criterion(predicted_indices.flatten(0, 2)[mask.flatten(0).to(torch.bool)],
                                  indices.flatten(0)[mask.flatten(0).to(torch.bool)].to(torch.long))
            losses.append(loss.item())
        _, predicted_indices = torch.max(predicted_indices.data, -1)
        predicted_indices = (predicted_indices * mask + indices * (~mask.to(torch.bool))).type(torch.int64)
        images_mask = (indices * (~mask.to(torch.bool))).type(torch.int64)
        self.plot_images(indices, show=False, save=f"{self.follow.path_samples}/original.png")
        self.plot_images(predicted_indices, show=False, save=f"{self.follow.path_samples}/reconstructed.png")
        self.plot_images(images_mask, show=False, save=f"{self.follow.path_samples}/masked.png")
        return losses

    def load(self, path: str = "", optimizer: bool = True):
        print("LOAD [", end="")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        if optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
        self.load_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"model: ok  | optimizer:{optimizer}  |  loss: {loss}  |  epoch: {self.load_epoch}]")

    def decode_vqvae(
            self,
            image_embeds
    ):
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))
        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h=h, w=w)
        images = self.vqvae.decoder(image_embeds.type(torch.FloatTensor).to("cuda"))
        return images
