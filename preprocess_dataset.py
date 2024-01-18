from src import preprocess_evaluation, EvaluationDataset
from src import VQVAE, SpeechVQVAE
import hydra
from omegaconf import DictConfig
import os
from src import size_model
import torch
import numpy as np
from sklearn.utils import shuffle

torch.cuda.empty_cache()

# ----------------------------------------------------------------------------------------------------------------
root = r"E:\Data\enterface database"
dataset_name = "enterface"
h5_path = r"E:\Data\enterface-preprocess\enterface-96-new.hdf5"
mae_path = r"checkpoint/CMDMAE/2023-4-3/21-3"
# ----------------------------------------------------------------------------------------------------------------


@hydra.main(config_path=f"{mae_path}/config_cmdmae", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    """ Data """
    dataset = EvaluationDataset(root=root,
                                speaker_retain_test=[],
                                dataset=dataset_name,
                                h5_path=h5_path
                                )



    """ Model: VQVAE"""
    vqvae_1 = VQVAE(**cfg.vqvae_1)
    vqvae_1.load(
        path_model=r"checkpoint/VQVAE/2023-1-25/10-27/model_checkpoint")  # checkpoint/VQVAE/2023-1-25/10-27/model_checkpoint
    size_model(vqvae_1, text="vqvae_1")

    vqvae_2 = SpeechVQVAE(**cfg.vqvae_2)
    vqvae_2.load(path_model=r"checkpoint/SPEECH_VQVAE/2023-3-1/9-8/model_checkpoint")
    size_model(vqvae_2, text="vqvae_2")

    preprocess = preprocess_evaluation(dataset=dataset,
                                       h5_path=h5_path,
                                       save_dataset=None,
                                       face_detector=False,
                                       dataset_preprocess=r"E:\Data\enterface-preprocess",
                                       vqvae_audio=vqvae_2, vqvae_visual=vqvae_1)



if __name__ == '__main__':
    main()

