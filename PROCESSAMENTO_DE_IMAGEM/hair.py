import cv2


def remove_hair(input_image, output_directory):
    src = cv2.imread(input_image)

    print(src.shape)
    cv2.imshow("original Image", src)

    # Convert the original image to grayscale
    grayScale = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    cv2.imshow("GrayScale", grayScale)
    #cv2.imwrite('filters/grayScale_sample1.jpg', grayScale, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1, (17, 17))

    # Perform the blackHat filtering on the grayscale image to find the
    # hair countours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    cv2.imshow("BlackHat", blackhat)
    #cv2.imwrite('filters/blackhat_sample1.jpg', blackhat, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # intensify the hair countours in preparation for the inpainting
    # algorithm
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    print(thresh2.shape)
    cv2.imshow("Thresholded Mask", thresh2)
    #cv2.imwrite('filters/thresholded_sample1.jpg', thresh2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # inpaint the original image depending on the mask
    dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)
    cv2.imshow("InPaint", dst)
    #cv2.imwrite(output_directory, dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


if __name__ == "__main__":
    remove_hair('./images/pele.jpg', './images/hair')
