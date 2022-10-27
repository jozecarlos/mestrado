import os
import sys

import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from operation import morphology
from segmentation import watershed
from segmentation import hough
from segmentation import adptative_threshold
from filters import median
from model import config
from model import image as imageModel


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


def moving_averages(image_path):
    img = cv.imread(image_path, 0)
    img = cv.medianBlur(img, 5)
    th = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

    plot(img, th, "Limiarização por média móvel")


def fourier_transaformation(image_path):
    print("Fourier Transformation")
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    dark_image_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(dark_image_grey, cmap='gray')
    plt.show()

    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(dark_image_grey))
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
    plt.show()

    bandlimit_index = int(1000 * dark_image_grey_fourier.size / 44100)

    for i in range(bandlimit_index + 1, len(dark_image_grey_fourier) - bandlimit_index):
        dark_image_grey_fourier[i] = 0

    real = dark_image_grey_fourier.real
    phases = dark_image_grey_fourier.imag

    # modify real part, put your modification here
    real_mod = real/3

    # create an empty complex array with the shape of the input image
    fft_img_shift_mod = np.empty(real.shape, dtype=complex)

    # insert real and phases to the new file
    fft_img_shift_mod.real = real_mod
    fft_img_shift_mod.imag = phases

    # reverse shift
    fft_img_mod = np.fft.ifftshift(fft_img_shift_mod)

    # reverse the 2D fourier transform
    img_mod = np.fft.ifft2(fft_img_mod)

    # using np.abs gives the scalar value of the complex number
    # with img_mod.real gives only real part. Not sure which is proper
    img_mod = np.abs(img_mod)

    # f_ishift = np.fft.ifftshift(dark_image_grey_fourier)
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(np.log(abs(img_mod)), cmap='gray')
    plt.show()


def question_01():
    fourier_transaformation('./images/cross_lines.jpg')
    # fourier_transaformation('./images/diagonal_lines.jpg')
    # fourier_transaformation('./images/horizontal_lines.png')
    # fourier_transaformation('./images/xadrez.png')


def question_02():
    morphology.erode('./images/circles.png', 10, True)
    morphology.dilate('./images/circles.png', 3, True)


def question_03():
    adptative_threshold.apply('./images/living_room.jpeg')
    #moving_averages('./images/living_room.jpeg')


def question_04():
    w = watershed.Watershed()
    # image = w.apply('./images/water_coins.jpeg')
    image = w.apply('./images/water_coins.jpeg')
    plt.imshow(image, cmap='Paired', interpolation='nearest')
    plt.show()


def question_05():
    w = hough.Hough()

    orig = cv2.imread('./images/avenida.png', cv2.IMREAD_COLOR)
    image = w.apply('./images/avenida.png')

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    ax[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.tight_layout()

    plt.imshow(image, cmap='Paired', interpolation='nearest')
    # plt.show()

    orig = cv2.imread('./images/xicara.jpg', cv2.IMREAD_COLOR)
    image = w.circles('./images/xicara.jpg')

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    ax[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.tight_layout()

    plt.imshow(image, cmap='Paired', interpolation='nearest')
    plt.show()


if __name__ == "__main__":
 question_01()
 question_02()
 question_03()
 question_04()
 question_05()
