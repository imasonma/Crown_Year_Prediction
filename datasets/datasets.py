from prefetch_generator import BackgroundGenerator
from datasets import Dataloader
from transforms import build_transform
from torch.utils.data import DataLoader


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def build_loader(config, phase):
    """build data loader for model training
    Args:
        config: the config parameters
        phase: the training or validating
    """

    # read datasets
    dimension = config.DATA.DIMENSION
    modal = config.DATA.MAIN_MODAL + config.DATA.SUB_MODAL
    transform = build_transform(config) if phase == "train" else None
    adjacent_layer = config.DATA.NER_LAYER
    # data loader parameters
    batch_size = config.DATA.BATCH_SIZE
    num_workers = config.DATA.NUM_WORKERS

    if dimension == 2:
        if phase == "train":
            file_dir_train = config.DATA.FILE_PATH_TRAIN
            train_dataset = DataLoader.Dataset(file_dir_train)
            train_dataset = Dataloader2d.MultiMedDatasets2D(
                file_dir_train, modal, transform, adjacent_layer, config = config
            )
            train_loader = DataLoaderX(
                train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
            )
            return train_loader
    else:
        raise "Only Support 2D or 3D Data dimension, check the config file"
