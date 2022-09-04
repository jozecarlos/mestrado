import numpy as np


def calc(config):
    buffer = np.zeros(256)
    for j in range(0, config.image.row):
        for k in range(0, config.image.col):
            buffer[(config.image.src[j, k])] += 1

    return 100*buffer/(config.image.row*config.image.col)
