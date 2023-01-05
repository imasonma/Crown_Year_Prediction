import os
import sys
import cc3d
from tqdm import tqdm
import numpy as np
from skimage import measure

import matplotlib.pyplot as plt


def analyse_image(image_dir):
    image_size = []

    filenames = os.listdir(image_dir)
    for idx in tqdm(filenames):
        image_path = image_dir + idx
        image = load_img_info(image_path)

        image_shape = image_array.shape
        image_shape = [image_shape[0] * spacing[0],
                       image_shape[1] * spacing[1]]
        image_size.append(image_shape)

    labels = ['y', 'x']
    plot_properties(image_size, 'size', labels, bins=80)


def plot_properties(properties, name, labels, bins=40):
    properties = np.array(properties)

    if len(labels) == 3:
        plt.figure()
        plt.subplot(131)
        plt.hist(properties[:, 0], bins=bins)
        plt.title('{} {}'.format(name, labels[0]))
        plt.subplot(132)
        plt.hist(properties[:, 1], bins=bins)
        plt.title('{} {}'.format(name, labels[1]))
        plt.subplot(133)
        plt.hist(properties[:, 2], bins=bins)
        plt.title('{} {}'.format(name, labels[2]))
        plt.show()

        print('{} mean is {}:{}, {}:{}, {}:{}'.format(name, labels[0], np.mean(properties[:, 0]),
                                                      labels[1], np.mean(properties[:, 1]),
                                                      labels[2], np.mean(properties[:, 2])))
        print('{} std is {}:{}, {}:{}, {}:{}'.format(name, labels[0], np.std(properties[:, 0]),
                                                     labels[1], np.std(properties[:, 1]),
                                                     labels[2], np.std(properties[:, 2])))
        print('{} max is {}:{}, {}:{}, {}:{}'.format(name, labels[0], np.max(properties[:, 0]),
                                                     labels[1], np.max(properties[:, 1]),
                                                     labels[2], np.max(properties[:, 2])))
        print('{} min is {}:{}, {}:{}, {}:{}'.format(name, labels[0], np.min(properties[:, 0]),
                                                     labels[1], np.min(properties[:, 1]),
                                                     labels[2], np.min(properties[:, 2])))
    elif len(labels) == 1:
        plt.figure()
        plt.hist(properties, bins=bins)
        plt.title('{} {}'.format(name, labels[0]))
        plt.show()
        print('{} mean is {}:{}'.format(name, labels[0], np.mean(properties)))
        print('{} std is {}:{}'.format(name, labels[0], np.std(properties)))
        print('{} max is {}:{}'.format(name, labels[0], np.max(properties)))
        print('{} min is {}:{}'.format(name, labels[0], np.min(properties)))
