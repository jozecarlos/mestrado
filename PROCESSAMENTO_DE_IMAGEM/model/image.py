import cv2


class MImage:

    def __init__(self, path):
        self.src = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.row, self.col = self.src.shape
