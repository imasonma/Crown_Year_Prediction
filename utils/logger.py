# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import sys
import logging
import functools
from termcolor import colored


def plot_image(writer, data, position='Train/input', step=0):
    data = data.as_tensor() if hasattr(data, 'as_tensor') else data
    batch_size = data.size()[0]
    writer.add_images(position, data[:batch_size // 2], step)
    return data

def plot_line(writer, data, position='Train/input', step=0):
    data = data.as_tensor() if hasattr(data, 'as_tensor') else data
    writer.add_scalar(position, data, step)
    return data

@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=""):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    color_fmt = (
        colored("[%(asctime)s %(name)s]", "green")
        + colored("(%(filename)s %(lineno)d)", "yellow")
        + ": %(levelname)s %(message)s"
    )

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            # logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
            logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(console_handler)

    # create file handlers
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(output_dir, f"log_rank{dist_rank}.txt"), mode="a"
    )
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    # # file_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    # logger.addHandler(file_handler)

    return logger


