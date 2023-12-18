from src import CMDMAE_Train, VoxcelebSequentialMultimodal, VoxcelebSequentialMultimodalH5, \
    VoxcelebSequentialMultimodalPT, SpeechVoxceleb_Static
from src import CMDMAE, VQVAE, SpeechVQVAE, h5_creation
import hydra
from omegaconf import DictConfig
import os
from src import size_model
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

torch.cuda.empty_cache()


@hydra.main(config_path="config_cmdmae", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    """ Data """
    training_vox = VoxcelebSequentialMultimodalH5(root=r"D:\These\data\Audio-Visual\voxceleb\test\video",
                                                  transform=transforms.Compose(
                                                      [transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
                                                       transforms.Resize(128)]),
                                                  train=True,
                                                  table_path=None,
                                                  h5_path_1=r"H5_temps/modality_1.hdf5",
                                                  h5_path_2=r"E:/cmd_mae/modality_2_test.hdf5"
                                                  )

    validation_vox = VoxcelebSequentialMultimodalH5(root=r"D:\These\data\Audio-Visual\voxceleb\test\video",
                                                    transform=transforms.Compose(
                                                        [transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
                                                         transforms.Resize(128)]),
                                                    train=False,
                                                    table_path=None,
                                                    h5_path_1=r"H5/modality_1_test.hdf5",
                                                    h5_path_2=r"H5/modality_2_test.hdf5"
                                                    )

    """ Model: VQVAE"""
    vqvae_1 = VQVAE(**cfg.vqvae_1)
    vqvae_1.load(path_model=r"checkpoint/VQVAE/2023-1-25/10-27/model_checkpoint")
    size_model(vqvae_1, text="vqvae_1")

    vqvae_2 = SpeechVQVAE(**cfg.vqvae_2)
    vqvae_2.load(path_model=r"checkpoint/SPEECH_VQVAE/2023-3-1/9-8/model_checkpoint")
    size_model(vqvae_2, text="vqvae_2")

    h5_creation(vqvae=vqvae_2,
                voxceleb=VoxcelebSequentialMultimodal(root=r"D:\These\data\Audio-Visual\voxceleb\train",
                                                      transform=transforms.Compose(
                                                          [transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
                                                           transforms.Resize(128),
                                                           transforms.CenterCrop(96)]),
                                                      train=True),
                dir_save="E:/cmd_mae/modality_2_train.hdf5",
                audio_bool=True)

    """ Model: MAE"""
    cmdmae = CMDMAE(**cfg.model,
                    alpha=(1.0, 1.0),
                    decoder_cross_attention=False,
                    encoder_cross_attention=False,
                    pos_embedding_trained=True,
                    vqvae_v_embedding=None,
                    vqvae_a_embedding=None,
                    mlp_ratio=2.0,
                    contrastive=False)
    size_model(cmdmae, text="cmdmae")


    """ Training """
    pretrain_vqvae = CMDMAE_Train(cmdmae, vqvae_1, vqvae_2,
                                  training_vox, validation_vox,
                                  tube_bool=True,
                                  config_training=cfg.train,
                                  gpu_monitor=True)
    # pretrain_vqvae.load(path="checkpoint/CMDMAE/2023-1-10/10-24/model_checkpoint")
    pretrain_vqvae.fit()


if __name__ == '__main__':
    main()
    # pe = pretrain_vqvae.model.decoder.pos_embedding.pe.cpu().detach().squeeze(1).numpy().T
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
    # pos = ax.imshow(pe, cmap="RdGy", extent=(1, pe.shape[1] + 1, pe.shape[0] + 1, 1))
    # plt.show()
    # vqvae_1.vq_vae._embedding.weight
