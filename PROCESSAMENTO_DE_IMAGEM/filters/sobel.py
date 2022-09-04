import sys

sys.path.insert(1, "{}{}".format(sys.path[1], '/util'))
sys.path.insert(1, "{}{}".format(sys.path[1], '/model'))

import numpy as np
import math as m


def filter(config):
    horz = np.zeros((config.kernel_size, config.kernel_size))
    horz[:, 0] = -1
    horz[int(config.kernel_size / 2), 0] = -2
    horz[:, (config.kernel_size - 1)] = 1
    horz[int(config.kernel_size / 2), (config.kernel_size - 1)] = 2

    vert = np.zeros((config.kernel_size, config.kernel_size))
    vert[0, :] = -1
    vert[0, int(config.kernel_size / 2)] = -2
    vert[(config.kernel_size - 1), :] = 1
    vert[(config.kernel_size - 1), int(config.kernel_size / 2)] = 2
    central = m.floor((config.kernel_size / 2))
    C = np.zeros((config.image.row + central * 2, config.image.col + central * 2))
    C[(0 + central):(config.image.row + central), (0 + central):(config.image.col + central)] = np.array(
        config.image.src)

    new_image = np.zeros(config.image.src.shape)
    somaHorz = 0
    somaVert = 0

    for image_row in range(0, config.image.row):
        for image_col in range(0, config.image.col):
            for kernel_row in range(0, config.kernel_size):
                for kernel_col in range(0, config.kernel_size):
                    somaHorz = (C[image_row + kernel_row, image_col + kernel_col] * horz[
                        kernel_row, kernel_col]) + somaHorz
                    somaVert = (C[image_row + kernel_row, image_col + kernel_col] * vert[
                        kernel_row, kernel_col]) + somaVert

            Ph = m.ceil((somaHorz / (config.kernel_size ** 2)))
            Pv = m.ceil((somaVert / (config.kernel_size ** 2)))
            somaHorz = 0
            somaVert = 0
            new_image[image_row, image_col] = np.sqrt(Ph ** 2 + Pv ** 2)

    return np.uint8(new_image)
