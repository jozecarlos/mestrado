import sys

sys.path.insert(1, "{}{}".format(sys.path[1], '/util'))
sys.path.insert(1, "{}{}".format(sys.path[1], '/model'))

import numpy as np
import math as m


def filter(config):
    central = m.floor((config.kernel_size / 2))
    clone_img_border = np.zeros((config.image.row + central * 2, config.image.col + central * 2))
    clone_img_border[(0 + central):(config.image.row + central), (0 + central):(config.image.col + central)] = np.array(
        config.image.src)

    buffer = [0 for _ in range(config.kernel_size * config.kernel_size)]
    new_image = np.zeros(config.image.src.shape)

    for image_row in range(0, config.image.row):
        for image_col in range(0, config.image.col):
            for kernel_row in range(0, config.kernel_size):
                for kernel_col in range(0, config.kernel_size):
                    buffer[(config.kernel_size * kernel_row + kernel_col)] = (
                    clone_img_border[image_row + kernel_row, image_col + kernel_col])

            buffer = np.sort(buffer)
            value = buffer[int(np.floor((config.kernel_size ** 2) / 2))]
            new_image[image_row, image_col] = value

    return np.uint8(new_image)
