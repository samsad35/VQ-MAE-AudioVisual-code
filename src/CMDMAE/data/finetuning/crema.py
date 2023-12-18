import os
from tqdm import tqdm
import glob
import pandas
import numpy as np
from pathlib import Path


class Crema:
    def __init__(self, root: Path = None, ext="flv"):
        self.root = root
        self.length = len(glob.glob(f"{root}/**/*.{ext}"))//2
        self.table = None
        self.emotion_map = dict(ANG=0, DIS=1, FEA=2, HAP=3, NEU=4, SAD=5)

    @staticmethod
    def __generator__(directory: Path):
        all_dir = os.listdir(directory)
        for d in all_dir:
            yield d, directory / d

    def generator(self):
        for name, path in self.__generator__(self.root):
            id = name.split(".")[0].split("_")[0]
            emotion = self.emotion_map[name.split(".")[0].split("_")[2]]
            yield id, name, path, emotion

    def generate_table(self):
        files_list = []
        id_list = []
        name_list = []
        emotion_list = []
        with tqdm(total=self.length, desc=f"Create table (CREMA-D): ") as pbar:
            for id, name, path, emotion in self.generator():
                files_list.append(path)
                id_list.append(id)
                emotion_list.append(emotion)
                name_list.append(name)
                pbar.update(1)
        self.table = pandas.DataFrame(np.array([id_list, name_list,  files_list, emotion_list]).transpose(),
                                      columns=['id', 'name', 'path', 'emotion'])


if __name__ == '__main__':
    vox = Crema(root=Path(r"E:\CREMA-D\VideoFlash"))
    vox.generate_table()
    print(vox.table['path'])

