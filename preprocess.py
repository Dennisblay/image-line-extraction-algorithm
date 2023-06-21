import cv2
import numpy as np


class PreProcess:

    def __init__(self, path=None):
        self.filepath = path

    def read_frame(self, filepath):
        if filepath is not None:
            self.filepath = filepath
            return cv2.imread(filepath)
        if self.filepath:
            return cv2.imread(self.filepath)
        return None

    def resize_by_half(self, image):
        return cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    # get grayscale image
    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    def denoise(self, image):
        return cv2.medianBlur(image, 5)

    # thresholding
    def threshold(self, image):
        return cv2.threshold(image, 5, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def adaptiveThreshold(self, image):
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 119, 1)

    # dilation
    def dilate(self, image, iterations=1):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=iterations)

    # erosion
    def erode(self, image, iterations=1):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=iterations)

    # opening - erosion followed by dilation
    def opening(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # canny edge detection
    def canny(self, image):
        return cv2.Canny(image, 100, 200)

    # save image
    def save_image_to_file(self, image, filename, file_extension=None):
        if file_extension is None:
            file_extension = self.filepath.split(".")[-1]
        cv2.imwrite(filename + file_extension, image)

    def show_frame(self, image):
        cv2.imshow("image", image)
        self.waitKey()

    def waitKey(self, delay=0):
        return cv2.waitKey(delay)
