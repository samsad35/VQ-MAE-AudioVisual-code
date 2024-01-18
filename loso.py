from src import preprocess_evaluation, EvaluationDataset, Classifier_Train
from src import CMDMAE, VQVAE, SpeechVQVAE
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
h5_path = r"E:\Data\enterface-preprocess\enterface-96.hdf5"
mae_path = r"checkpoint/CMDMAE/2023-4-3/21-3"
num_classes = 6
query2emo = False
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

    all_id = shuffle(np.unique(np.array(dataset.table["id"])))
    Total_folds = len(list(all_id))
    accuracy_epoch = []
    f1_epoch = []

    for num_fold in range(Total_folds):
        speaker_retain_test = []
        print(f"Fold number: {num_fold + 1}/{Total_folds}")
        speaker_retain_test.append(all_id[num_fold])
        # print(speaker_retain_test)
        data_train = EvaluationDataset(root=root,
                                       speaker_retain_test=speaker_retain_test,
                                       frames_per_clip=50,
                                       train=True,
                                       dataset=dataset_name,
                                       h5_path=h5_path
                                       )

        data_test = EvaluationDataset(root=root,
                                      speaker_retain_test=speaker_retain_test,
                                      frames_per_clip=50,
                                      train=False,
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

        """ VQ-MAE-AUDIOVISUAL"""
        cmdmae = CMDMAE(**cfg.model,
                        alpha=(1.0, 1.0),
                        pos_embedding_trained=True,
                        vqvae_v_embedding=None,
                        vqvae_a_embedding=None,
                        mlp_ratio=2.0)
        size_model(cmdmae, text="cmdmae")
        cmdmae.load(path_model=f"{mae_path}/model_checkpoint")
        # cmdmae = cmdmae.requires_grad_(False)

        """ Run classifier """
        pretrain_classifier = Classifier_Train(cmdmae,
                                               data_train,
                                               data_test,
                                               config_training=cfg.train,
                                               follow=True,
                                               query2emo=query2emo,
                                               pooling="attention",
                                               num_classes=num_classes)  # pooling: mean, attention
        # pretrain_classifier.load(path="checkpoint/CLASSIFIER/2023-1-23/10-31/model_checkpoint")
        accuracy, f1 = pretrain_classifier.fit()
        accuracy_epoch.append(accuracy)
        f1_epoch.append(f1)
        print("-" * 50)
        break
    print(f"Accuracy final: {np.mean(accuracy_epoch)}")
    print(f"F1 final: {np.mean(f1_epoch)}")


if __name__ == '__main__':
    main()
    # pe = cmdmae.encoder.modality_emb_v.cpu().detach().squeeze(1).numpy().T
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
    # pos = ax.imshow(pe, cmap="RdGy", extent=(1, pe.shape[1] + 1, pe.shape[0] + 1, 1))
    # plt.show()
