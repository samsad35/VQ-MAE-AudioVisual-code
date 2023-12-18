import h5py
from ...model import VQVAE, SpeechVQVAE
from ..sequential import VoxcelebSequentialMultimodal
from ...tools import read_video_decord
from tqdm import tqdm
import torch
import librosa
import numpy as np
from ...tools import load_audio_from_video, to_spec

torch.cuda.empty_cache()

win_length = int(64e-3 * 16000)
spec_parameters = dict(n_fft=1024,
                       hop=int(0.625 / 2 * win_length),
                       # hop=int(0.625 * win_length),
                       win_length=win_length)


def load_wav(file: str):
    wav, sr = librosa.load(path=file, sr=16000)
    wav = librosa.to_mono(wav.transpose())
    wav = wav / np.max(np.abs(wav))
    return wav


def h5_creation(vqvae,
                voxceleb,
                dir_save: str,
                audio_bool: bool = False):
    vox = voxceleb.vox
    file_h5 = h5py.File(dir_save, 'a')
    vqvae.to('cuda')
    with tqdm(total=len(voxceleb.table)) as pbar:
        for part, id, ytb_id, name, file in vox.generator():
            pbar.update(1)
            pbar.set_description(f"ID: {id}, ytb_id: {ytb_id}, name: {name}")
            # Get indices for each file .mp4
            if file_h5.get(f'/{part}/{id}/{ytb_id}/{name}'):
                continue
            if audio_bool:
                data = load_wav(file=file)
                data, _ = to_spec(data, spec_parameters)
                # indices = (data ** 2).transpose()
                # print(indices.shape)
                data = torch.from_numpy((data ** 2).transpose()).type(torch.FloatTensor).unsqueeze(1)
            else:
                data = read_video_decord(file_path=file).transpose((0, 3, 1, 2))
                data = torch.from_numpy(data) / 255.0
                data = voxceleb.transform(data)
            try:
                data = data.to("cuda")
                indices = vqvae.get_codebook_indices(data).cpu().detach().numpy()
            except RuntimeError:
                data = data.to('cpu')
                if data.shape[0] > 1000:
                    data = data[:500]
                vqvae.to('cpu')
                indices = vqvae.get_codebook_indices(data).numpy()
                vqvae.to('cuda')
            # Create the path in H5 file:
            if not file_h5.get(f'/{part}'):
                file_h5.create_group(f'/{part}')
            if not file_h5.get(f'/{part}/{id}'):
                file_h5.create_group(f'/{part}/{id}')
            if not file_h5.get(f'/{part}/{id}/{ytb_id}'):
                file_h5.create_group(f'/{part}/{id}/{ytb_id}')
            group_temp = file_h5[f'/{part}/{id}/{ytb_id}']
            # Save indices in H5 file
            image_h5 = group_temp.create_dataset(name=name, data=indices, dtype="int32")
            image_h5.attrs.create('id', id)

    # Close h5 file
    file_h5.flush()
    file_h5.close()
