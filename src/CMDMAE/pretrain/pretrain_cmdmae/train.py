from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import numpy as np
from ...base import Train
from ...model import VQVAE, CMDMAE, SpeechVQVAE
from ...tools import griffin_lim, Monitor
import matplotlib.pyplot as plt
from .contrastive_loss import ContrastiveLoss
from .follow_up_cmdmae import Follow
import math
from einops import rearrange
from torchvision.utils import make_grid
import kornia as K
from scipy.io.wavfile import write
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .idr_torch import IDR
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
torch.cuda.empty_cache()


class CMDMAE_Train(Train):
    def __init__(self, cmdmae: CMDMAE, vqvae_1: VQVAE, vqvae_2: SpeechVQVAE,
                 training_data: Dataset, validation_data: Dataset,
                 tube_bool: bool = False,
                 config_training: dict = None,
                 multigpu_bool: bool = False,
                 gpu_monitor: bool = False):
        super().__init__()
        if multigpu_bool:
            self.idr = IDR()
            dist.init_process_group(backend='nccl',
                                    init_method='env://',
                                    world_size=self.idr.size,
                                    rank=self.idr.rank)
            torch.cuda.set_device(self.idr.local_rank)

        self.device = torch.device(config_training['device'])
        """ Model """
        self.model = cmdmae
        self.vqvae_1 = vqvae_1
        self.vqvae_2 = vqvae_2
        self.model = self.model.to(self.device)
        self.vqvae_1.to(self.device)
        self.vqvae_2.to(self.device)
        self.contrastive_bool = self.model.contrastive
        if multigpu_bool:
            self.model = DDP(self.model, device_ids=[self.idr.local_rank], find_unused_parameters=True)

        """ Dataloader """
        if multigpu_bool:
            train_sampler = torch.utils.data.distributed.DistributedSampler(training_data,
                                                                            num_replicas=self.idr.size,
                                                                            rank=self.idr.rank,
                                                                            shuffle=True)
            self.training_loader = torch.utils.data.DataLoader(dataset=training_data,
                                                               batch_size=config_training[
                                                                              'batch_size'] // self.idr.size,
                                                               shuffle=False,
                                                               num_workers=config_training['num_workers'],
                                                               pin_memory=True,
                                                               drop_last=True,
                                                               sampler=train_sampler)
            val_sampler = torch.utils.data.distributed.DistributedSampler(validation_data,
                                                                          num_replicas=self.idr.size,
                                                                          rank=self.idr.rank,
                                                                          shuffle=True)
            self.validation_loader = torch.utils.data.DataLoader(dataset=validation_data,
                                                                 batch_size=config_training[
                                                                                'batch_size'] // self.idr.size,
                                                                 shuffle=False,
                                                                 num_workers=0,
                                                                 pin_memory=True,
                                                                 drop_last=True,
                                                                 sampler=val_sampler,
                                                                 prefetch_factor=2)
        else:
            self.training_loader = DataLoader(training_data, batch_size=config_training['batch_size'], shuffle=True,
                                              num_workers=config_training['num_workers'], drop_last=True,)
            self.validation_loader = DataLoader(validation_data, batch_size=config_training['batch_size'], shuffle=True,
                                                num_workers=0, drop_last=True)

        """ Optimizer """
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=config_training['lr']*config_training['batch_size']/256,
                                           betas=(0.9, 0.95),
                                           weight_decay=config_training["weight_decay"])
        lr_func = lambda epoch: min((epoch + 1) / (config_training["warmup_epoch"] + 1e-8),
                                    0.5 * (math.cos(epoch / config_training["total_epoch"] * math.pi) + 1))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_func, verbose=True)

        """ Loss """
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.nceLoss = ContrastiveLoss(batch_size=config_training['batch_size'], temperature=0.5)

        """ Config """
        self.config_training = config_training
        self.load_epoch = 0
        self.step_count = 0
        self.parameters = dict()
        self.h5_bool = training_data.h5_bool
        self.multigpu_bool = multigpu_bool
        self.tube_bool = tube_bool

        """ Follow """
        self.follow = Follow("cmdmae", dir_save=r"checkpoint", multigpu_bool=multigpu_bool)
        if gpu_monitor:
            self.gpu_monitor = Monitor(delay=60)

    @staticmethod
    def to_tube(input, input_image: bool = True, size_patch=4):
        if input_image:
            c1 = c2 = int(math.sqrt(input.shape[-1])/size_patch)
            t1 = input.shape[1]//5
            input = rearrange(input, 'b (t1 t2) (c1 l1 c2 l2) -> b (t1 c1 c2) (l1 l2 t2)',
                              t1=t1, t2=5, c1=c1, c2=c2, l1=size_patch, l2=size_patch)
        else:
            c1 = int(input.shape[-1]/size_patch)
            t1 = input.shape[1]//5
            input = rearrange(input, 'b (t1 t2) (c1 l1) -> b (t1 c1) (l1 t2)', t1=t1, t2=5, c1=c1, l1=size_patch)
        return input

    @staticmethod
    def inverse_tuple(input, input_image: bool = True, size_patch=4):
        if input_image:
            input = rearrange(input, 'b (t1 c1 c2) (l1 l2 t2) -> b (t1 t2) (c1 l1 c2 l2)',
                              t1=10, t2=5, c1=6, c2=6, l1=size_patch, l2=size_patch)
        else:
            input = rearrange(input, 'b (t1 c1) (l1 t2) -> b (t1 t2) (c1 l1)', t1=10, t2=5, c1=16, l1=size_patch)
        return input


    def one_epoch(self):
        self.model.train()
        losses = []
        contrastives = []
        for batch_idx, (modality_1, modality_2) in enumerate(tqdm(iter(self.training_loader))):
            self.optimizer.zero_grad()
            self.step_count += 1
            modality_1 = modality_1.to(self.device)  # non_blocking=True
            modality_2 = modality_2.to(self.device)  # non_blocking=True
            if not self.h5_bool:
                # Modality: 1
                batch = modality_1.shape[0]
                modality_1 = rearrange(modality_1, 'b t c h w -> (b t) c h w')
                indices_1 = self.vqvae_1.get_codebook_indices(modality_1)
                indices_1 = torch.reshape(indices_1, (batch, -1, 16 * 16))
                # Modality: 2
                modality_2 = rearrange(modality_2, 'b t c h w -> (b t) c h w')
                indices_2 = self.vqvae_2.get_codebook_indices(modality_2)
                indices_2 = torch.reshape(indices_2, (batch, -1, 16 * 16))
            else:
                indices_1, indices_2 = modality_1, modality_2
            if self.tube_bool:
                indices_1 = self.to_tube(indices_1, input_image=True, size_patch=4)
                indices_2 = self.to_tube(indices_2, input_image=False, size_patch=4)
            # To CMDMAE + Losses
            if self.contrastive_bool:
                predicted_indices_1, mask_1, cls_1, predicted_indices_2, mask_2, cls_2 = self.model(indices_1, indices_2)
                contrastive_loss = self.nceLoss(cls_1, cls_2)
                contrastive_loss = 0.0 if torch.isnan(contrastive_loss) else contrastive_loss
            else:
                predicted_indices_1, mask_1, predicted_indices_2, mask_2 = self.model(indices_1, indices_2)
                contrastive_loss = 0.0

            # Losses
            loss_1 = self.criterion(predicted_indices_1.flatten(0, 2)[mask_1.flatten(0).to(torch.bool)],
                                    indices_1.flatten(0)[mask_1.flatten(0).to(torch.bool)].to(torch.long))
            loss_2 = self.criterion(predicted_indices_2.flatten(0, 2)[mask_2.flatten(0).to(torch.bool)],
                                    indices_2.flatten(0)[mask_2.flatten(0).to(torch.bool)].to(torch.long))
            loss_1 = 0.0 if torch.isnan(loss_1) else loss_1
            loss_2 = 0.0 if torch.isnan(loss_2) else loss_2
            loss = loss_1 + loss_2 + contrastive_loss
            losses.append(loss.item())
            contrastives.append((contrastive_loss.item()))
            loss.backward()
            self.optimizer.step()
        return losses, contrastives

    def fit(self):
        for e in range(self.load_epoch, self.config_training["total_epoch"]):
            if self.multigpu_bool:
                self.training_loader.sampler.set_epoch(e)
                self.validation_loader.sampler.set_epoch(e)
            losses, contrastives = self.one_epoch()
            self.lr_scheduler.step()
            if e % 1 == 0:
                losses_val = self.eval()
                avg_loss_train = sum(losses) / len(losses)
                avg_loss_contrastive = sum(contrastives) / len(contrastives)
                avg_loss_val = sum(losses_val) / len(losses_val)
                if self.multigpu_bool:
                    model_parameter = self.model.module.state_dict()
                else:
                    model_parameter = self.model.state_dict()
                self.parameters = dict(model=model_parameter,
                                       optimizer=self.optimizer.state_dict(),
                                       scheduler=self.lr_scheduler.state_dict(),
                                       epoch=e,
                                       loss=avg_loss_train)
                print(
                    f'In epoch {e}, average traning loss is {avg_loss_train}. '
                    f'and average validation loss is {avg_loss_val}'
                    f'and contrastive loss is {avg_loss_contrastive}')
                self.follow(epoch=e, loss_train=avg_loss_train, loss_validation=avg_loss_val,
                            parameters=self.parameters)
            self.gpu_monitor.stop()

    @staticmethod
    def save_wav(indices, save: str = None, vqvae: SpeechVQVAE = None):
        audio = vqvae.decode(indices[0])
        # np.save(file=save, arr=np.sqrt(torch.transpose(audio.squeeze(1), 0, 1).cpu().detach().numpy()))
        signal = griffin_lim(np.sqrt(torch.transpose(audio.squeeze(1), 0, 1).cpu().detach().numpy()))
        write(save, 16000, signal)

    @staticmethod
    def plot_images(indices, show: bool = True, save: str = None, vqvae: VQVAE = None):
        images = vqvae.decode(indices[0])
        plt.figure(figsize=(15, 15))
        out: torch.Tensor = make_grid(images + 0.5, nrow=10, padding=10)
        out_np: np.array = K.tensor_to_image(out)
        plt.imshow(out_np)
        plt.axis('off')
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)
            plt.close()

    def eval(self):
        torch.cuda.empty_cache()
        self.model.eval()
        losses = []
        for modality_1, modality_2 in tqdm(iter(self.validation_loader)):
            modality_1 = modality_1.to(self.device, non_blocking=True)
            modality_2 = modality_2.to(self.device, non_blocking=True)
            if not self.h5_bool:
                # Modality: 1
                batch = modality_1.shape[0]
                modality_1 = rearrange(modality_1, 'b t c h w -> (b t) c h w')
                indices_1 = self.vqvae_1.get_codebook_indices(modality_1)
                indices_1 = torch.reshape(indices_1, (batch, -1, 16 * 16))
                # Modality: 2
                modality_2 = rearrange(modality_2, 'b t c h w -> (b t) c h w')
                indices_2 = self.vqvae_2.get_codebook_indices(modality_2)
                indices_2 = torch.reshape(indices_2, (batch, -1, 16 * 16))
            else:
                indices_1, indices_2 = modality_1, modality_2
            if self.tube_bool:
                indices_1 = self.to_tube(indices_1, input_image=True, size_patch=4)
                indices_2 = self.to_tube(indices_2, input_image=False, size_patch=4)
            with torch.no_grad():
                # To CMDMAE + Losses
                if self.contrastive_bool:
                    predicted_indices_1, mask_1, cls_1, predicted_indices_2, mask_2, cls_2 = self.model(indices_1,
                                                                                                        indices_2)
                    contrastive_loss = self.nceLoss(cls_1, cls_2)
                    contrastive_loss = 0.0 if torch.isnan(contrastive_loss) else contrastive_loss
                else:
                    predicted_indices_1, mask_1, predicted_indices_2, mask_2 = self.model(indices_1, indices_2)
                    contrastive_loss = 0.0
                loss_1 = self.criterion(predicted_indices_1.flatten(0, 2)[mask_1.flatten(0).to(torch.bool)],
                                        indices_1.flatten(0)[mask_1.flatten(0).to(torch.bool)].to(torch.long))
                loss_2 = self.criterion(predicted_indices_2.flatten(0, 2)[mask_2.flatten(0).to(torch.bool)],
                                        indices_2.flatten(0)[mask_2.flatten(0).to(torch.bool)].to(torch.long))
                loss_1 = 0.0 if torch.isnan(loss_1) else loss_1
                loss_2 = 0.0 if torch.isnan(loss_2) else loss_2
                loss = loss_1 + loss_2 + contrastive_loss
                losses.append(loss.item())
        # Plot image for modality 1
        _, predicted_indices_1 = torch.max(predicted_indices_1.data, -1)
        predicted_indices_1 = (predicted_indices_1 * mask_1 + indices_1 * (~mask_1.to(torch.bool))).type(torch.int64)
        images_mask_1 = (indices_1 * (~mask_1.to(torch.bool))).type(torch.int64)
        if self.tube_bool:
            indices_1 = self.inverse_tuple(indices_1, input_image=True, size_patch=4)
            predicted_indices_1 = self.inverse_tuple(predicted_indices_1, input_image=True, size_patch=4)
            images_mask_1 = self.inverse_tuple(images_mask_1, input_image=True, size_patch=4)
        self.plot_images(indices_1, show=False, save=f"{self.follow.path_samples}/1_original.png", vqvae=self.vqvae_1)
        self.plot_images(indices_1, show=False, save=f"{self.follow.path_samples}/1_original.svg", vqvae=self.vqvae_1)
        self.save_animation(indices_1, vqvae=self.vqvae_1, video_name=f"{self.follow.path_samples}/1_original.mp4")
        self.plot_images(predicted_indices_1, show=False, save=f"{self.follow.path_samples}/1_reconstructed.png",
                         vqvae=self.vqvae_1)
        self.plot_images(predicted_indices_1, show=False, save=f"{self.follow.path_samples}/1_reconstructed.svg",
                         vqvae=self.vqvae_1)
        self.save_animation(predicted_indices_1, vqvae=self.vqvae_1,
                            video_name=f"{self.follow.path_samples}/1_reconstructed.mp4")
        self.plot_images(images_mask_1, show=False, save=f"{self.follow.path_samples}/1_masked.png", vqvae=self.vqvae_1)
        self.plot_images(images_mask_1, show=False, save=f"{self.follow.path_samples}/1_masked.svg", vqvae=self.vqvae_1)
        self.save_animation(images_mask_1, vqvae=self.vqvae_1, video_name=f"{self.follow.path_samples}/1_masked.mp4")
        # Load wav for modality 2
        _, predicted_indices_2 = torch.max(predicted_indices_2.data, -1)
        predicted_indices_2 = (predicted_indices_2 * mask_2 + indices_2 * (~mask_2.to(torch.bool))).type(torch.int64)
        images_mask_2 = (indices_2 * (~mask_2.to(torch.bool))).type(torch.int64)
        if self.tube_bool:
            indices_2 = self.inverse_tuple(indices_2, input_image=False, size_patch=4)
            predicted_indices_2 = self.inverse_tuple(predicted_indices_2, input_image=False, size_patch=4)
            images_mask_2 = self.inverse_tuple(images_mask_2, input_image=False, size_patch=4)
        self.save_wav(indices_2, save=f"{self.follow.path_samples}/2_original.wav", vqvae=self.vqvae_2)
        self.save_wav(predicted_indices_2, save=f"{self.follow.path_samples}/2_reconstructed.wav", vqvae=self.vqvae_2)
        self.save_wav(images_mask_2, save=f"{self.follow.path_samples}/2_masked.wav", vqvae=self.vqvae_2)
        return losses

    def load(self, path: str = "", optimizer: bool = True):
        print("LOAD [", end="")
        checkpoint = torch.load(path)
        if self.multigpu_bool:
            self.model.module.load_state_dict(checkpoint['model'])  # load checkpoint for multi-GPU
        else:
            checkpoint = torch.load(path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
        if optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
        self.load_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"model: ok  | optimizer:{optimizer}  |  loss: {loss}  |  epoch: {self.load_epoch}]")

    @staticmethod
    def save_animation(indices, vqvae: VQVAE, video_name: str = "temps/animation.mp4"):
        images = vqvae.decode(indices[0])
        images = images.cpu().detach().numpy()
        fps = 25
        fig = plt.figure(figsize=(3, 3))
        ax = plt.axes()
        plt.axis('off')
        grp = ax.imshow(np.transpose(images[0], (1, 2, 0)) + 0.5, 'gray')

        def update(frame_number):
            image = images[frame_number]
            grp.set_array(np.transpose(image, (1, 2, 0)) + 0.5)
            return grp,

        anim = FuncAnimation(fig, update, frames=images.shape[0], interval=1000 / fps, repeat=False)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, bitrate=1800, metadata=dict(artist='VQ-MAE-AV'))
        anim.save(video_name, writer=writer)
        plt.close()

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
