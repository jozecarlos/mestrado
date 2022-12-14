import sys

sys.path.insert(1, "{}{}".format(sys.path[1], '/util'))
sys.path.insert(1, "{}{}".format(sys.path[1], '/model'))

import numpy as np
import math as m


def filter(config):
    kernel = np.ones((config.kernel_size, config.kernel_size))
    kernel[int(config.kernel_size / 2), int(config.kernel_size / 2)] = -1 * (np.sum(config.kernel_size) - 1)
    central = m.floor((config.kernel_size / 2))
    clone_img_border = np.zeros((config.image.row + central * 2, config.image.col + central * 2))
    clone_img_border[(0 + central):(config.image.row + central), (0 + central):(config.image.col + central)] = np.array(
        config.image.src)

    new_image = np.zeros(config.image.src.shape)
    count = 0

    for image_row in range(0, config.image.row):
        for image_col in range(0, config.image.col):
            for kernel_row in range(0, config.kernel_size):
                for kernel_col in range(0, config.kernel_size):
                    count = (clone_img_border[image_row + kernel_row, image_col + kernel_col] * kernel[kernel_row, kernel_col]) + count

            value = m.ceil((count / (config.kernel_size * config.kernel_size)))
            count = 0
            new_image[image_row, image_col] = value

    return np.uint8(new_image)
