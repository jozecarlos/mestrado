import sys

sys.path.insert(1, "{}{}".format(sys.path[1], '/util'))
sys.path.insert(1, "{}{}".format(sys.path[1], '/model'))

import numpy as np
import math as m


def gaussian_kernel(h1, h2):
    x, y = np.mgrid[0:h2, 0:h1]
    x = x - h2 / 2
    y = y - h1 / 2
    sigma = 0.5
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def filter(config):
    kernel = gaussian_kernel(config.kernel_size, config.kernel_size)
    central = m.floor((config.kernel_size / 2))
    C = np.zeros((config.image.row + central * 2, config.image.col + central * 2))
    C[(0 + central):(config.image.row + central), (0 + central):(config.image.col + central)] = np.array(
        config.image.src)

    new_image = np.zeros(config.image.src.shape)
    count = 0

    for image_row in range(0, config.image.row):
        for image_col in range(0, config.image.col):
            for kernel_row in range(0, config.kernel_size):
                for kernel_col in range(0, config.kernel_size):
                    count = (C[image_row + kernel_row, image_col + kernel_col] * kernel[kernel_row, kernel_col]) + count

            #value = m.ceil(count)
            value = m.ceil((count / (config.kernel_size * config.kernel_size)))
            count = 0
            new_image[image_row, image_col] = value

    return np.uint8(new_image)
