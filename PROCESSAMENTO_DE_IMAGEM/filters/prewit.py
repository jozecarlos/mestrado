import sys

sys.path.insert(1, "{}{}".format(sys.path[1], '/util'))
sys.path.insert(1, "{}{}".format(sys.path[1], '/model'))

import numpy as np
import math as m


def filter(config):
    horz = np.zeros((config.kernel_size, config.kernel_size))
    horz[:, 0] = -1
    horz[:, (config.kernel_size - 1)] = 1

    vert = np.zeros((config.kernel_size, config.kernel_size))
    vert[0, :] = -1
    vert[(config.kernel_size - 1), :] = 1

    central = m.floor((config.kernel_size / 2))
    clone_img_border = np.zeros((config.image.row + central * 2, config.image.col + central * 2))
    clone_img_border[(0 + central):(config.image.row + central), (0 + central):(config.image.col + central)] = np.array(
        config.image.src)

    new_image = np.zeros(config.image.src.shape)
    count_h = 0
    count_v = 0

    for image_row in range(0, config.image.row):
        for image_col in range(0, config.image.col):
            for kernel_row in range(0, config.kernel_size):
                for kernel_col in range(0, config.kernel_size):
                    count_h = (clone_img_border[image_row + kernel_row, image_col + kernel_col] * horz[
                        kernel_row, kernel_col]) + count_h
                    count_v = (clone_img_border[image_row + kernel_row, image_col + kernel_col] * vert[
                        kernel_row, kernel_col]) + count_v

            result_h = m.ceil((count_h / (config.kernel_size ** 2)))
            result_v = m.ceil((count_v / (config.kernel_size ** 2)))

            count_h = 0
            count_v = 0

            new_image[image_row, image_col] = np.sqrt(result_h ** 2 + result_v ** 2)

    return np.uint8(new_image)
