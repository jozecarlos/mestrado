import cv2
import numpy as np


class Hough(object):

    def apply(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR) # road.png is the filename
        # Convert the image to gray-scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the edges in the image using canny detector
        edges = cv2.Canny(gray, 50, 200)
        # Detect points that form a line
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=10, maxLineGap=250)
        # Draw lines on the image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        # Show result
        return img

    def circles(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # Convert to gray-scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Blur the image to reduce noise
        img_blur = cv2.medianBlur(gray, 5)
        # Apply hough transform on the image
        circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 10, img.shape[0]/64, param1=200, param2=10, minRadius=5, maxRadius=30)
        # Draw detected circles
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw outer circle
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw inner circle
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        return img
