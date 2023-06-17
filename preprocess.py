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

    # get grayscale image
    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    def denoise(self, image):
        return cv2.medianBlur(image, 5)

    # thresholding
    def threshold(self, image):
        return cv2.threshold(image, 12, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def adaptiveThreshold(self, image):
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 119 , 1)

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

    # skew correction
    def deskew(self, image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated

    # template matching
    def match_template(self, image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    # save image
    def save_image_to_file(self, image, filename, file_extension=None):
        if file_extension is None:
            file_extension = self.filepath.split(".")[-1]
        cv2.imwrite(filename + file_extension, image)

    def show_frame(self, image):
        cv2.imshow(r"frame", image)
        self.waitKey()

    def waitKey(self, delay=0):
        cv2.waitKey(delay)
