import glob
import random
import os

import numpy as np
import torch

from datasets.consis import ConsisLearingDataset


# from transforms import Transforms
class MonoMedDataSets2D(torch.utils.data.Dataset):
    """
    A dataloader to read Mono-Modal Dataset

    For example, typical input data can be a list of dictionaries:
    label path: self.file_dir/{file_mode}/label/*.npy
    input path: self.file_dir/{file_mode}/{data_type}/*.npy
    """

    def __init__(
        self,
        file_dir,
        transform=None,
        file_mode=None,
        data_type=None,
        adjacent_layer=None,
    ) -> None:
        """
        Args:
            file_dir: Directory of the file to read
            transform: a transform function to transform
            file_mode: the file mode to read, must be your own root file
            data_type: the data type to read, must be your own data type
            adjacent_layer: the adjacent layer to read, you could set 2.5d model by the parameter
        """
        self.file_dir = os.path.join(file_dir, file_mode)
        self.label = sorted(glob.glob(os.path.join(self.file_dir, "*/label/*")))
        self.inputs = sorted(glob.glob(os.path.join(self.file_dir, f"*/{data_type}/*")))
        self.transform = transform
        self.adjacent_layer = adjacent_layer

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx):
        inputs = self.Normalization(self.ReadData(self.inputs[idx]))
        label = self.noise_label(self.ReadData(self.label[idx]))
        sample = {
            "image": inputs,
            "label": label,
        }
        if self.transform:
            sample = self.transform(sample)
            # sample = self.transform(sample)
        return sample

    def OverSampling(self, label, idx):
        """Over Sampling the data by the label
        Args:
            label(np.ndarray): label data
            idx: the index of the label
        Return:
            idx: the index of the label
            label: the label of the data
        """
        whether_label = torch.sum(torch.gt(label, 1).float()).bool()
        while whether_label == 0:
            idx += 1
            if idx >= len(self.file_dir):
                idx = random.randint(0, len(self.label) - 1)
            label = self.Normalization(np.load(self.label[idx]).astype(float))
            whether_label = torch.sum(torch.sum(label))
        return idx, label

    def Normalization(self, data, dim_threshold: int = 3) -> torch.Tensor:
        """
        make a torch Tensor
        Args:
            data(np.ndarray): input data
        Return:
            torch.Tensor
        """
        assert (
            len(data.shape) <= dim_threshold
        ), "data must be 3-dimensional or below 3-dimensional."
        if len(data.shape) == dim_threshold:
            return torch.from_numpy(data)
        return torch.unsqueeze(torch.from_numpy(data), 0)

    def ReadData(self, filename) -> np.ndarray:
        """
        To Read Data using numpy
        Args:
            filename(str): filename of the file
        Return:
            np.ndarray: if adjacent_layer is not None,
            return the adjacent layer data size:adjacent_layer,W,H. else return 1,W,H
        """
        if self.adjacent_layer is not None:
            assert self.adjacent_layer >= 0, "adjacent_layer must be >= 0"
            filename_name = filename.split("/")[-1].split(".")[0]
            filename_dir = os.path.dirname(filename)
            assert filename_name.isdigit(), "Filename must be a number"
            if (
                int(filename_name) >= self.adjacent_layer
                and int(filename_name)
                <= len(glob.glob(os.path.join(filename_dir, "*")))
                - self.adjacent_layer
                - 1
            ):
                return self.ReadMultiData(filename_dir, int(filename_name))
            elif int(filename_name) < self.adjacent_layer:
                return self.ReadMultiData(
                    filename_dir, int(filename_name) + self.adjacent_layer
                )
            else:
                return self.ReadMultiData(
                    filename_dir, int(filename_name) - self.adjacent_layer - 1
                )
        return np.load(filename).astype(float)

    def ReadMultiData(self, filedir, filename):
        data = []
        for i in range(
            filename - self.adjacent_layer, filename + self.adjacent_layer + 1
        ):
            data.append(
                np.load(os.path.join(filedir, f"{str(i)}.npy")).astype(float)[
                    np.newaxis, :
                ]
            )
        return np.vstack(data).astype(float)


class MonoMedDataSets2DTest(MonoMedDataSets2D):
    """
    A 3D datasets dataloder to be better calculating the dice coefficient.
    """

    def __init__(self, file_dir, file_mode=None, data_type=None):
        self.file_dir = os.path.join(file_dir, file_mode)
        self.label = sorted(glob.glob(os.path.join(self.file_dir, "*/label/")))
        self.inputs = sorted(glob.glob(os.path.join(self.file_dir, f"*/{data_type}/")))
        self.adjacent_layer = None

    def __getitem__(self, idx):
        label_list = [
            os.path.join(self.label[idx], f"{str(i)}.npy")
            for i in range(len(glob.glob(os.path.join(self.label[idx], "*"))))
        ]
        input_list = [
            os.path.join(self.inputs[idx], f"{str(i)}.npy")
            for i in range(len(glob.glob(os.path.join(self.label[idx], "*"))))
        ]

        label = []
        inputs = []
        for idxs, i in enumerate(label_list):
            label.append(self.ReadData(i)[np.newaxis, np.newaxis, :])
            inputs.append(self.ReadData(input_list[idxs])[np.newaxis, np.newaxis, :])
        label = self.Normalization(np.vstack(label), 4)
        inputs = self.Normalization(np.vstack(inputs), 4)
        sample = {
            "image": inputs,
            "label": label,
        }

        return sample


class MultiMedDatasets2D(MonoMedDataSets2DTest):
    """
    A dataloader to read Multi-Modal Dataset

    For example, typical input data can be a list of dictionaries:
    label path: self.file_dir/{file_mode}/label/*.npy
    input path: self.file_dir/{file_mode}/{data_type}/*.npy
    """

    def __init__(self, file_dir, data_type, transform=None, adjacent_layer=None, config=None):
        self.file_dir = file_dir
        self.label = sorted(glob.glob(os.path.join(self.file_dir, "*/label/*")))

        self.data_type = data_type
        self.transform = transform
        self.adjacent_layer = adjacent_layer

        self.consis = config.TRAINING.TTA.CONSISLEARNING
        if config and config.TRAINING.TTA.CONSISLEARNING:
            self.consis_trans = ConsisLearingDataset(config)

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx):
        sample = {}
        label = self.Normalization(np.load(self.label[idx]).astype(float))
        if random.random() < 0.2:
            idx, label = self.OverSampling(label, idx)

        for i in self.data_type:
            if i == 'img':
                sample[i] = self.Normalization(self.ReadData(self.MultiModalPath(i)[idx]))
            else:
                sample[i] = self.Normalization(self.ReadData(self.MultiModalPath(i)[idx]))
        sample["label"] = label

        if self.consis:
            sample = self.consis_trans(sample)

        if self.transform:
            sample = self.transform(sample)
        return sample

    def MultiModalPath(self, modal: str = None) -> list:
        return sorted(glob.glob(os.path.join(self.file_dir, f"*/{modal}/*")))


class MultiMedDatasets2DTest(MonoMedDataSets2DTest):
    def __init__(self, file_dir, data_type=None, transform=None, adjacent_layer=None):
        self.file_dir = file_dir
        self.label = sorted(glob.glob(os.path.join(self.file_dir, "*/label/")))
        self.adjacent_layer = adjacent_layer
        self.data_type = data_type
        self.transform = transform

    def __getitem__(self, idx):
        sample = {}
        sample["label"] = self.Read3DData(self.label[idx], 'label')
        for i in self.data_type:
            sample[i] = self.Read3DData(self.label[idx].replace("label", i))
        if self.transform:
            sample = self.transform(sample, "test")

        sample['path'] = self.label[idx]
        return sample

    def Read3DData(self, modal_path, read_type = 'img'):
        data_list = [
            os.path.join(modal_path, f"{str(i)}.npy")
            for i in range(len(glob.glob(os.path.join(modal_path, "*"))))
        ]
        data = []
        if read_type != 'img':
            for _, i in enumerate(data_list):
                new_data = np.load(i)
                if len(new_data.shape) == 2:
                    new_data = new_data[np.newaxis, np.newaxis, :]
                else:
                    new_data = new_data[np.newaxis, :]
                data.append(new_data)
        else:
            for _, i in enumerate(data_list):
                new_data = self.ReadData(i)
                if len(new_data.shape) == 2:
                    new_data = new_data[np.newaxis, np.newaxis, :]
                else:
                    new_data = new_data[np.newaxis, :]
                data.append(new_data)
        # data = self.Normalization(np.vstack(data), 4)
        data = self.Normalization(np.concatenate(data, axis=0), 4)#.permute(0,1,3,2)

        return data


class MultiMedDataSets2DInf(MonoMedDataSets2D):
    def __init__(self, file_dir, data_type=None, transform=None, adjacent_layer=None, weather3d=False):
        self.file_dir = file_dir
        if isinstance(data_type, (list, tuple)):
            self.main_type = data_type[0]
        self.label = sorted(glob.glob(os.path.join(self.file_dir, f"*/{self.main_type}/")))
        self.adjacent_layer = adjacent_layer
        self.data_type = data_type
        self.transform = transform
        self.weather3d = weather3d

    def __getitem__(self, idx):
        sample = {}
        for i in self.data_type:
            if self.weather3d:
                sample[i] = self.Read3DData(self.label[idx].replace(self.main_type, i))
                sample['path'] = self.label[idx]
            else:
                sample[i] = self.Normalization(self.ReadData(self.MultiModalPath(i)[idx].replace(self.main_type, i)))
        if self.transform:
            sample = self.transform(sample, "test")


if __name__ == "__main__":
    dataset = MultiMedDatasets2DTest(
        file_dir="/raid0/xxx/Y064/Dataset", file_mode="NPY_val", data_type=["CT", "T1"]
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for idx, sample in enumerate(dataloader):
        print(sample["T1"].shape)
