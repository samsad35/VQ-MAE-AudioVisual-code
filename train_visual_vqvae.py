from src import VQVAE_Train, VQVAE
from dalle_pytorch import DiscreteVAE
from src import Voxceleb_Static
import torchvision.transforms as transforms
import hydra
from omegaconf import DictConfig
import os


@hydra.main(config_path="config_vqvae", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    """ Data """
    vox = Voxceleb_Static(root=cfg.dataset.parameters.root,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
                                                        transforms.Resize(cfg.dataset.parameters.resize)
                                                        ]),
                          train=True
                          )


    """ Model """
    # model = DiscreteVAE(**cfg.model)
    model = VQVAE(**cfg.model)

    """ Training """
    pretrain_vqvae = VQVAE_Train(model, vox, vox, config_training=cfg.train)
    # pretrain_vqvae.load(path=r"checkpoint\VQVAE\2023-1-1\15-48/model_checkpoint")
    pretrain_vqvae.fit()


if __name__ == '__main__':
    main()
