import argparse
import datetime
import os
import time
import random

import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import monai
import deepspeed
import SimpleITK as sitk

from tensorboardX import SummaryWriter
from torch import optim

from datasets import build_loader
from losses import DiceCoefficient, SegLoss, MeanVolume
from model import build_model, collect_params
from utils import get_config, load_checkpoint, save_checkpoint, save_nii_data
from utils.logger import create_logger, plot_image, plot_line

from utils_main import _build_optimizer

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser("training and evaluation script", add_help=False)
    parser.add_argument(
        "--cfg", type=str, required=True, metavar="FILE", help="path to config file"
    )

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return config


class Trainer(object):
    def __init__(self, config):

        self.config = config
        output_dir = os.path.join(
            config.ENVIRONMENT.EXPERIMENT_PATH, config.ENVIRONMENT.NAME
        )
        self.logger = create_logger(output_dir)

        self.logger.info(f"Creating model:{config.MODEL.NAME}")
        self.model = build_model(config)
        if config.ENVIRONMENT.CUDA:
            self.model.cuda()

        if config.ENVIRONMENT.PHASE == "train":
            self.logger.info(f"Loading {config.TRAINING.OPTIMIZER.METHOD} optimizer")
            self.optimizer = _build_optimizer(self.config, self.model)
            if config.ENVIRONMENT.AMP_OPT_LEVEL != "O0":
                self.model, self.optimizer = amp.initialize(
                    self.model,
                    self.optimizer,
                    opt_level=config.ENVIRONMENT.AMP_OPT_LEVEL,
                )

        # logging the number of model parameters and flops
        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.logger.info(f"number of params: {n_parameters}")


        self.lr_scheduler = (
            _get_lr_scheduler(config, self.optimizer) if config.ENVIRONMENT.PHASE == "train" else None
        )

        # TensorBoard logger
        experiment_dir = os.path.join(
            config.ENVIRONMENT.EXPERIMENT_PATH, config.ENVIRONMENT.NAME
        )
        self.train_writer = SummaryWriter(log_dir=os.path.join(experiment_dir, "train"))
        self.val_writer = SummaryWriter(log_dir=os.path.join(experiment_dir, "val"))

    def _create_train_data(self):
        self.train_data_loader = build_loader(self.config, phase="train")
        self.validate_data_loader = build_loader(self.config, phase="val")

    def train_one_epoch(self, epoch):
        batch_size = self.config.DATA.BATCH_SIZE
        self.model.train()
        dice_val = 0
        main_modal = self.config.DATA.MAIN_MODAL
        sub_modal = self.config.DATA.SUB_MODAL
        label_name = self.config.DATA.LABEL[0]
        for _idx, data in enumerate(self.train_data_loader):
            if self.config.ENVIRONMENT.CUDA:
                image = image.cuda(non_blocking=True)[0]
                label = label.cuda(non_blocking=True)
            pass

if __name__ == "__main__":
    config = parse_option()
    if config.ENVIRONMENT.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    SEED = config.ENVIRONMENT.SEED
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    cudnn.benchmark = True
    cudnn.deterministic = True
    cudnn.enabled = True

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.ENVIRONMENT.NUM_GPU)
    segmentor = Trainer(config)
    segmentor.do_train()
