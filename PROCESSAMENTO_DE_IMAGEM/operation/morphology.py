import numpy as np
import cv2
import json
from matplotlib import pyplot as plt


def convert_binary(image_src, thresh_val):
    color_1 = 255
    color_2 = 0
    initial_conv = np.where((image_src <= thresh_val), image_src, color_1)
    final_conv = np.where((initial_conv > thresh_val), initial_conv, color_2)
    return final_conv


def binarize(image_file, thresh_val=127):
    image_src = cv2.imread(image_file, 0)
    image_b = convert_binary(image_src=image_src, thresh_val=thresh_val)
    return image_b


def dilate(image_path, dilatation_level=3, with_plot=False):
    img = cv2.imread(image_path, 0)
    kernel = np.ones((dilatation_level, dilatation_level), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    if with_plot:
        plot(img, img_dilation, "Dilated - {}".format(dilatation_level))
        plt.show()
        return True
    return img_dilation


def erode(image_path, erosion_level=3, with_plot=False):
    erosion_level = 3 if erosion_level < 3 else erosion_level

    structuring_kernel = np.full(shape=(erosion_level, erosion_level), fill_value=255)
    image_src = binarize(image_file=image_path)

    orig_shape = image_src.shape
    pad_width = erosion_level - 2

    # pad the matrix with `pad_width`
    image_pad = np.pad(array=image_src, pad_width=pad_width, mode='constant')
    pimg_shape = image_pad.shape
    h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])

    # sub matrices of kernel size
    flat_submatrices = np.array([
        image_pad[i:(i + erosion_level), j:(j + erosion_level)]
        for i in range(pimg_shape[0] - h_reduce) for j in range(pimg_shape[1] - w_reduce)
    ])

    # condition to replace the values - if the kernel equal to submatrix then 255 else 0
    image_erode = np.array([255 if (i == structuring_kernel).all() else 0 for i in flat_submatrices])
    image_erode = image_erode.reshape(orig_shape)

    if with_plot:
        plot(image_src, image_erode, "Eroded - {}".format(erosion_level))
        return True
    return image_erode


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
