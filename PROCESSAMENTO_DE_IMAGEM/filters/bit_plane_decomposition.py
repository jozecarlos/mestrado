import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Enter the path of the picture to get 8 bitmaps
def get_bitmaps(image):
    bit_extraction = []
    bit_images = []
    for i in range(8):
        bit_extraction.append(np.ones(image.shape, dtype=np.uint8) * pow(2, i))
    for i in range(8):
        bit_images.append(cv.bitwise_and(image, bit_extraction[i]))
    return bit_images


def decomposite():
    lenna = cv.imread('../images/pele_sem_pelos.jpg', cv.IMREAD_COLOR)
    lenna = lenna[:, :, [2, 1, 0]]
    gray_lenna = cv.cvtColor(lenna, cv.COLOR_BGR2GRAY)
    gray_bit_images = get_bitmaps(gray_lenna)
    color_bit_images = get_bitmaps(lenna)

    plt.figure(figsize=(12, 12))
    plt.suptitle('gray image bitmap', fontsize=16)
    plt.subplot(3, 3, 1), plt.axis('off'), plt.title('melanoma'), plt.imshow(gray_lenna, cmap='gray')
    for i in range(8):
        plt.subplot(3, 3, i + 2), plt.axis('off'), plt.title('bit_img_' + str(i)), plt.imshow(gray_bit_images[i],
                                                                                              cmap='gray')
    plt.show()

    plt.figure(figsize=(12, 12))
    plt.suptitle('color image bitmap', fontsize=16)
    plt.subplot(3, 3, 1), plt.axis('off'), plt.title('melanoma'), plt.imshow(lenna)
    for i in range(8):
        plt.subplot(3, 3, i + 2), plt.axis('off'), plt.title('bit_img_' + str(i)), plt.imshow(color_bit_images[i])
    plt.show()


if __name__ == "__main__":
    decomposite()
