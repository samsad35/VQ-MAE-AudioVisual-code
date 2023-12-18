from ...tools import read_video_moviepy, face_detection_mctnn, face_detection_cascade, read_video_decord, to_spec
from .dataset_evaluation import EvaluationDataset
from ...model import SpeechVQVAE, VQVAE
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import torch
import torchvision.transforms as transforms
import librosa
import h5py


def load_wav(file: str):
    wav, sr = librosa.load(path=file, sr=16000)
    wav = librosa.to_mono(wav.transpose())
    wav = wav / np.max(np.abs(wav))
    return wav


def save_animation(images, video_name: str = "temps/animation.mp4"):
    fps = 25
    fig = plt.figure(figsize=(3, 3))
    ax = plt.axes()
    plt.axis('off')
    grp = ax.imshow(images[0])

    def update(frame_number):
        image = images[frame_number]
        grp.set_array(image)
        return grp,

    anim = FuncAnimation(fig, update, frames=images.shape[0], interval=1000 / fps, repeat=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=1800, metadata=dict(artist='VQ-MAE-AV'))
    anim.save(video_name, writer=writer)
    plt.close()


def preprocess_evaluation(dataset: EvaluationDataset,
                          h5_path: str = None,
                          face_detector: bool = True,
                          save_dataset: str = None,
                          dataset_preprocess: str = None,
                          vqvae_audio: SpeechVQVAE = None,
                          vqvae_visual: VQVAE = None):
    data = dataset.data
    if save_dataset is not None:
        path_root = Path(save_dataset)
    transform = transforms.Compose(
        [transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
         transforms.CenterCrop(256),
         transforms.Resize(128),  # 128 #224
         transforms.CenterCrop(96)])  # 96 #192

    win_length = int(64e-3 * 16000)
    spec_parameters = dict(n_fft=1024,
                           hop=int((0.625 / 2.0) * win_length),
                           win_length=win_length)
    vqvae_visual.to("cuda")
    vqvae_audio.to("cuda")
    if h5_path is not None:
        file_h5 = h5py.File(h5_path, 'a')
    with tqdm(total=len(dataset.table)) as pbar:
        for id, name, path, emotion in data.generator():
            pbar.update(1)
            pbar.set_description(f"ID: {id}, name: {name}")

            if h5_path is not None:
                if file_h5.get(f'/{id}/audio_{name}'):
                    continue

            """ Create new dataset with face_detector if True """
            if save_dataset is not None:
                path_id = path_root / id
                path_id.mkdir(exist_ok=True)
                path_name = path_id / name
                if path_name.exists():
                    continue
            if face_detector and (save_dataset is not None):
                clip = read_video_moviepy(file_path=str(path), fps=25)
                frames = clip.iter_frames()
                sequences_frames = []
                for frame in frames:
                    try:
                        sequences_frames.append(face_detection_cascade(frame, resize=(256, 256)))
                    except:
                        continue
            if save_dataset is not None:
                try:
                    save_animation(np.array(sequences_frames), video_name=path_name)
                except:
                    print(path_name)
                    continue

            """ VQ-VAE """
            try:
                # Audio
                audio = load_wav(str(path))
                audio, phase = to_spec(audio, spec_parameters)
                audio = torch.from_numpy((audio ** 2).transpose()).type(torch.FloatTensor).unsqueeze(1)
                audio = audio.to("cuda")
                indices_audio = vqvae_audio.get_codebook_indices(audio).cpu().detach().numpy()

                # Visual
                print(f"{dataset_preprocess}/{id}/{name}")
                video = read_video_decord(f"{dataset_preprocess}/{id}/{name}").transpose((0, 3, 1, 2))
                video = torch.from_numpy(video) / 255.0
                video = transform(video)
                # plt.imshow(video[0].numpy().transpose((1, 2, 0)))
                # plt.show()
                video = video.to("cuda")
                indices_video = vqvae_visual.get_codebook_indices(video).cpu().detach().numpy()
            except:
                print(path_name)
                continue

            """ H5 creation """
            if h5_path is not None:
                if not file_h5.get(f'/{id}'):
                    file_h5.create_group(f'/{id}')
                group_temp = file_h5[f'/{id}']
                # Save indices in H5 file
                group_temp.create_dataset(name=f"audio_{name}", data=indices_audio, dtype="int32")
                group_temp.create_dataset(name=f"visual_{name}", data=indices_video, dtype="int32")
        # Close h5 file
        file_h5.flush()
        file_h5.close()
