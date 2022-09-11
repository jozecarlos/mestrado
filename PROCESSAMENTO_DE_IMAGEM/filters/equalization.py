import numpy as np


def calc(config):
    image = np.array(config.image.src)
    matrix = np.zeros((config.image.row, config.image.col))
    for row in range(0, config.image.row):
        for col in range(0, config.image.col):
            matrix[row, col] = np.ceil(255*(
                    (image[row, col] - image.min()) /
                    (image.max() - image.min())))

    return np.uint8(matrix)
