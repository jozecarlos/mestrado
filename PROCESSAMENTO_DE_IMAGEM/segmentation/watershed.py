import cv2
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.feature import peak_local_max


class Watershed(object):

    def apply(self, image_path):
        img = cv2.imread(image_path)
        b,g,r = cv2.split(img)
        rgb_img = cv2.merge([r,g,b])

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # noise removal
        kernel = np.ones((2,2),np.uint8)
        #opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(closing,kernel,iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)

        # Threshold
        ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1

        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        markers = cv2.watershed(img,markers)
        img[markers == -1] = [255,0,0]

        plt.subplot(421),plt.imshow(rgb_img)
        plt.title('Imagem de Entrada'), plt.xticks([]), plt.yticks([])
        plt.subplot(422),plt.imshow(thresh, 'gray')
        plt.title("Otsu's Limiarização"), plt.xticks([]), plt.yticks([])

        plt.subplot(423),plt.imshow(closing, 'gray')
        plt.title("Convolução kernel 2x2"), plt.xticks([]), plt.yticks([])
        plt.subplot(424),plt.imshow(sure_bg, 'gray')
        plt.title("Dilatação"), plt.xticks([]), plt.yticks([])

        plt.subplot(425),plt.imshow(dist_transform, 'gray')
        plt.title("Transformação pela Distancia"), plt.xticks([]), plt.yticks([])
        plt.subplot(426),plt.imshow(sure_fg, 'gray')
        plt.title("Limiarização"), plt.xticks([]), plt.yticks([])

        plt.subplot(427),plt.imshow(unknown, 'gray')
        plt.title("Detecção de Bordas"), plt.xticks([]), plt.yticks([])

        plt.subplot(428),plt.imshow(img, 'gray')
        plt.title("Resultado do Watershed"), plt.xticks([]), plt.yticks([])

        plt.tight_layout()
        plt.show()
