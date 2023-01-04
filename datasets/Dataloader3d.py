import os
import numpy as np
import glob
import random

# from transforms import Transforms
from datasets import Dataloader2d


class MedDataSets3D(Dataloader2d.MultiMedDatasets2DTest):
    def __init__(self, file_dir: str = None, data_type=None, transforms=None):
        self.file_dir = file_dir
        self.label = sorted(glob.glob(os.path.join(self.file_dir, "*/label/")))
        self.adjacent_layer = None
        # self.data_type = [i for i in data_type.replace(' ','').split(',')]
        self.data_type = data_type

    def __getitem__(self, idx):
        sample = {}
        sample["label"] = self.Read3DData(self.label[idx])
        for i in self.data_type:
            sample[f"{i}"] = self.Read3DData(self.label[idx].replace("label", i))
        sample = self.RandCropLayer(sample)

        return sample

    def Read3DData(self, modal_path):
        data_list = [
            os.path.join(modal_path, f"{str(i)}.npy")
            for i in range(len(glob.glob(os.path.join(modal_path, "*"))))
        ]
        data = []
        for idx, i in enumerate(data_list):
            data.append(self.ReadData(i)[np.newaxis, :])
        data = self.Normalization(np.vstack(data), 4)
        return data

    def RandCropLayer(self, sample, num_layer = 16):
        d_shape = sample["label"].shape[1]
        start_layer = random.randint(0, d_shape - num_layer - 1)
        for i in sample.keys():
            sample[i] = sample[i][:, start_layer: start_layer + num_layer, :]
        return sample


class MedDataSetsTrue3D(Dataloader2d.MultiMedDatasets2DTest):
    def __init__(self, file_dir: str = None, data_type=None, transforms=None):
        self.file_dir = file_dir
        self.label = sorted(glob.glob(os.path.join(self.file_dir, "*/label*")))
        self.data_type = data_type

    def __getitem__(self, idx):
        sample = {}
        sample["label"] = self.Normalization(np.load(self.label[idx]), 4)
        for i in self.data_type:
            sample[f"{i}"] = self.Normalization(np.load(self.label[idx].replace("label", i)), 4)

        return sample
