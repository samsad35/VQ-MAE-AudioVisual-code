from src import Speech_VQVAE_Train, SpeechVQVAE
from src import SpeechVoxceleb_Static
import hydra
from omegaconf import DictConfig
import os


@hydra.main(config_path="config_speech_vqvae", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    """ Data """
    vox = SpeechVoxceleb_Static(root=cfg.dataset.parameters.root, train=True,
                                spec_parameters=cfg.dataset.spec_parameters)

    """ Model """
    model = SpeechVQVAE(**cfg.model)

    """ Training """
    pretrain_vqvae = Speech_VQVAE_Train(model, vox, vox, config_training=cfg.train)
    # pretrain_vqvae.load(path=r"checkpoint/SPEECH_VQVAE/2022-12-24/20-57/model_checkpoint")
    pretrain_vqvae.fit()


if __name__ == '__main__':
    main()
