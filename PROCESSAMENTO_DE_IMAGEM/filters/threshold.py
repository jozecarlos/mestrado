import numpy as np


def apply(config, limiar):
    image = np.array(config.image.src)
    limiar_min = np.min(image)
    limiar_max = np.max(image)
    new_image = np.zeros(image.shape)

    for row in range(0, config.image.row):
        for col in range(0, config.image.col):
            if image[row, col] > limiar:
                new_image[row, col] = limiar_max
            else:
                new_image[row, col] = limiar_min

    return np.uint8(new_image)


def multi(config, multi_limiar=[], multi_range=[]):
    image = np.array(config.image.src)
    D = np.zeros(image.shape)
    if len(multi_limiar) == 2:
        T2 = multi_limiar[1]
        T1 = multi_limiar[0]
        Gmin = 0
        Gmed = 127
        Gmax = 255

    if len(multi_limiar) == 3:
        T3 = multi_limiar[2]
        T2 = multi_limiar[1]
        T1 = multi_limiar[0]
        Gmin = 0
        Gmed1 = multi_range[0]
        Gmed2 = multi_range[1]
        Gmax = 255

    for j in range(0, config.image.row):
        for k in range(0, config.image.col):
            if len(multi_limiar) == 3:
                if image[j, k] > T3:
                    D[j, k] = Gmax
                elif A[j, k] <= T3 and image[j, k] > T2:
                    D[j, k] = Gmed2
                elif image[j, k] <= T2 and image[j, k] > T1:
                    D[j, k] = Gmed1
                elif image[j, k] <= T1:
                    D[j, k] = Gmin

            elif len(multi_limiar) == 2:
                if image[j, k] > T2:
                    D[j, k] = Gmax
                elif image[j, k] <= T2 and image[j, k] > T1:
                    D[j, k] = Gmed
                elif image[j, k] <= T1:
                    D[j, k] = Gmin

    return np.uint8(D)
