import os
from tqdm import tqdm
import glob
import pandas
import numpy as np
from pathlib import Path


class Enterface:
    def __init__(self, root: Path = None, ext="avi"):
        self.root = root
        self.length = len(glob.glob(f"{root}/**/**/**/*.{ext}"))
        self.table = None

    @staticmethod
    def __generator__(directory: Path):
        all_dir = os.listdir(directory)
        for d in all_dir:
            yield d, directory / d

    def generator(self):
        for id, id_root in self.__generator__(self.root):
            for emotion, emotion_root in self.__generator__(id_root):
                for sentence, sentence_root in self.__generator__(emotion_root):
                    if sentence.split(".")[-1] == "db":
                        continue
                    if os.path.isfile(sentence_root):
                        yield id, sentence, sentence_root, emotion
                        continue
                    for name, path in self.__generator__(sentence_root):
                        if name.split(".")[-1] == "db":
                            continue
                        yield id, name, path, emotion
                # for name, path in self.__generator__(id_root):
                    # if name.split(".")[0].split("-")[0] != "01":  # video + audio
                    #     continue
                    # emotion = int(name.split(".")[0].split("-")[2])-2
                    # # level = int(name.split(".")[0].split("-")[3])-1
                    # yield id, name, path, emotion

    def generate_table(self):
        files_list = []
        id_list = []
        name_list = []
        emotion_list = []
        with tqdm(total=self.length, desc=f"Create table (Enterface): ") as pbar:
            for id, name, path, emotion in self.generator():
                files_list.append(path)
                id_list.append(id)
                emotion_list.append(emotion)
                name_list.append(name)
                pbar.update(1)
        self.table = pandas.DataFrame(np.array([id_list, name_list,  files_list, emotion_list]).transpose(),
                                      columns=['id', 'name', 'path', 'emotion'])


if __name__ == '__main__':
    vox = Enterface(root=Path(r"E:\Data\enterface database"))
    vox.generate_table()
    print(vox.table["path"])

