from ...model import VQVAE
from ..sequential import VoxcelebSequentialMultimodal
from ...tools import read_video_decord
from tqdm import tqdm
import torch
from pathlib import Path



def get_data(file, voxceleb, vqvae):
    data = read_video_decord(file_path=file).transpose((0, 3, 1, 2))
    data = torch.from_numpy(data) / 255.0
    data = voxceleb.transform(data)
    indices = vqvae.get_codebook_indices(data)
    return indices


def process(file, voxceleb, vqvae, dir_save, part, id, ytb_id, name):
    indices = get_data(file, voxceleb, vqvae)
    path_part = dir_save / part
    path_part.mkdir(exist_ok=True)

    path_id = path_part / id
    path_id.mkdir(exist_ok=True)

    path_ytb_id = path_id / ytb_id
    path_ytb_id.mkdir(exist_ok=True)
    torch.save(indices, f'{path_ytb_id / name}.pt')


def write_dataset(vqvae: VQVAE, voxceleb: VoxcelebSequentialMultimodal, dir_save: str):
    vox = voxceleb.vox
    dir_save = Path(dir_save)
    dir_save.mkdir(exist_ok=True)
    with tqdm(total=len(voxceleb.table)) as pbar:
        for part, id, ytb_id, name, file in vox.generator():
            pbar.update(1)
            pbar.set_description(id)
            process(file, voxceleb, vqvae, dir_save, part, id, ytb_id, name)
