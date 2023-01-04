import glob
import random
import os

import numpy as np
import imageio.v3 as iio
import torch

from monai.data import PILReader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.image_reader = PILReader()
        self.patient_path = ''
        pass

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx):
        pass

    def _get_image(self, data_path):
        return self.image_reader.read(data_path)

    def _get_video(self, data_path):
        return iio.imread(data_path, plugin="pyav")
