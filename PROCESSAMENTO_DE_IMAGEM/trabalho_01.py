import os
import sys
import time

from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from model import image
from model import config
from filters import median
from filters import mean
from filters import gaussian
from filters import laplace
from filters import prewit
from filters import sobel
from filters import histogram
from filters import equalization
from filters import threshold


def plot_image(original, legend_1, new, legend_2):
    plt.subplot(121), plt.imshow(original, 'gray')
    plt.subplot(121).set_title(legend_1)

    plt.subplot(122), plt.imshow(new, 'gray')
    plt.subplot(122).set_title(legend_2)
    plt.show()


def apply_median(config):
    print("Median Filter")

    start = time.time()
    result = median.filter(config)
    end = time.time()

    print("O tempo de execução do programa acima é:",(end-start) * 10**3, "ms")
    plot_image(image.src, 'Imagem original', result, 'Img com Filtro Mediano')


def apply_mean(config):
    print("Mean Filter")

    start = time.time()
    result = mean.filter(config)
    end = time.time()

    print("O tempo de execução do programa acima é:",(end-start) * 10**3, "ms")
    plot_image(image.src, 'Imagem original', result, 'Imagem com Filtro Medio')


def apply_gaussian(config):
    print("Gaussian Filter")

    start = time.time()
    result = gaussian.filter(config)
    end = time.time()

    print("O tempo de execução do programa acima é:",(end-start) * 10**3, "ms")
    plot_image(image.src, 'Imagem original', result, 'Imagem com Filtro Gaussiano')


def apply_laplace(config):
    print("Laplace Filter")

    start = time.time()
    result = laplace.filter(config)
    end = time.time()

    print("O tempo de execução do programa acima é:",(end-start) * 10**3, "ms")
    plot_image(image.src, 'Imagem original', result, 'Imagem com Filtro Laplaciano')


def apply_prewit(config):
    print("Prewit Filter")

    start = time.time()
    result = prewit.filter(config)
    end = time.time()

    print("O tempo de execução do programa acima é:",(end-start) * 10**3, "ms")
    plot_image(image.src, 'Imagem original', result, 'Imagem com Filtro Prewit')


def apply_sobel(config):
    print("Sobel Filter")

    start = time.time()
    result = sobel.filter(config)
    end = time.time()

    print("O tempo de execução do programa acima é:",(end-start) * 10**3, "ms")
    plot_image(image.src, 'Imagem original', result, 'Imagem com Filtro Sobel')


def limiar(config):
    print("Limiarização")

    start = time.time()
    result = threshold.apply(config, 139)
    end = time.time()

    print("O tempo de execução do programa acima é:",(end-start) * 10**3, "ms")
    plot_image(image.src, 'Imagem original', result, 'Imagem com Limiar 139')


def multi_limiar(config):
    print("Multi Limiarização")

    multi_limiar = np.array([5, 250])
    multi_range = np.array([127])
    result = threshold.multi(config, multi_limiar, multi_range)

    plot_image(image.src, 'Imagem original', result, 'Imagem com Limiar 139')


def calc_equalization(config):
    print("Equalization")
    eq = equalization.calc(config)
    hist = histogram.calc(config)

    plt.figure(1)
    plt.imshow(config.image.src, 'gray')
    plt.title('Imagem Original')

    plt.figure(2)
    plt.stem(hist)
    plt.title('Equalização da imagem')
    plt.show()


def calc_histogram(config):
    print("Histogram Calculation")
    hist = histogram.calc(config)

    plt.figure(1)
    plt.imshow(config.image.src, 'gray')
    plt.title('Imagem Original')

    plt.figure(2)
    plt.stem(hist)
    plt.title('Histograma da imagem')
    plt.show()


if __name__ == "__main__":
    image = image.MImage('./images/lena.png')
    config = config.MConfig(image, 3)

    apply_median(config)
    apply_mean(config)
    apply_gaussian(config)
    apply_laplace(config)
    apply_prewit(config)
    apply_sobel(config)
    calc_equalization(config)
    calc_histogram(config)
    limiar(config)
    multi_limiar(config)
