import os
import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from operation import morphology


def plot(image_src, img_final, label):
    cmap_val = 'gray'
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))

    ax1.axis("off")
    ax1.title.set_text('Original')

    ax2.axis("off")
    ax2.title.set_text(label)

    ax1.imshow(image_src, cmap=cmap_val)
    ax2.imshow(img_final, cmap=cmap_val)
    plt.show()


def watershed():


def moving_averages(image_path):
    img = cv.imread(image_path, 0)
    img = cv.medianBlur(img, 5)
    th = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

    plot(img, th, "Limiarização por média móvel")


def fourier_transaformation(image_path):
    print("Fourier Transformation")
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    dark_image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(dark_image_grey, cmap='gray')
    plt.show()

    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(dark_image_grey))
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
    plt.show()


def question_01():
    fourier_transaformation('./images/cross_lines.jpg')
    fourier_transaformation('./images/diagonal_lines.jpg')
    fourier_transaformation('./images/horizontal_lines.png')
    fourier_transaformation('./images/xadrez.png')


def question_02():
    morphology.erode('./images/circles.png', 3, True)
    morphology.dilate('./images/circles.png', 5, True)


def question_03():
    moving_averages('./images/living_room.jpeg')


if __name__ == "__main__":
    # question_01()
    # question_02()
    question_03()
