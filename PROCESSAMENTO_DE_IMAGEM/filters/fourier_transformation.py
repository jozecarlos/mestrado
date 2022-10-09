import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



if __name__ == "__main__":

    image = cv.imread('../images/pele.jpg', cv.IMREAD_COLOR)
    dark_image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(dark_image_grey, cmap='gray')
    plt.show()

    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(dark_image_grey))
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
    plt.show()
