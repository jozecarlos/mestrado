import cv2
import matplotlib.pyplot as plt


def apply(image_path):

    image = cv2.imread(image_path)

    #Otu's method requires greyscale images and blurring helps
    #both accentuate bi-modal colors, but also removes some noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    ret, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
    print(f'Threshold: {ret}')

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[1].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    plt.legend('Normal Threshold')
    plt.tight_layout()
    plt.show()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply adaptive thresholding
    mask = cv2.adaptiveThreshold(blurred,
                                 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY,
                                 31,
                                 10)

    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[1].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.legend('Adaptive Threshold')
    plt.show()
