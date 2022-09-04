import numpy as np


def calc(config):
    image = np.array(config.image.src)
    matrix = np.zeros((config.image.row, config.image.col))
    for j in range(0, config.image.row):
        for k in range(0, config.image.col):
            matrix[j, k] = np.ceil(255*(
                    (image[j, k] - image.min()) /
                    (image.max() - image.min())))

    return np.uint8(matrix)
