import cv2
import numpy as np


class MImage:

    def __init__(self, path):
        if not isinstance(path, np.ndarray):
            self.src = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            self.src = path
        self.row, self.col = self.src.shape
