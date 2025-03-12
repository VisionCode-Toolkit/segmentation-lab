import cv2
import numpy as np
from scipy.signal import correlate2d

from classes.filter import Filters



class Canny_detector():
    def __init__(self, output_image_viewer):
        self.output_image_viewer = output_image_viewer
        self.filter = Filters(self.output_image_viewer)

    def apply_canny_detector(self, kernel_size, sigma, low_thresh, high_tresh):
        # canny detector can be done in 5 steps
        # make sure the img is gray
        if len(self.output_image_viewer.current_image.modified_image.shape) == 3:
            self.output_image_viewer.current_image.transfer_to_gray_scale()

        # filter the img
        self.output_image_viewer.current_image.modified_image = self.filter.apply_filters(sigma, kernel_size)
        # obtain total change (gradient) and theta
        total_gradient, theta = self.calculate_gradient()

        # obtain resultant image from nms
        resultant_img = self.non_max_suppression(total_gradient,theta)

        # thresholding
        thresholded_img = self.double_thresholding(resultant_img, low_thresh, high_tresh)

    def kernels_for_gradient(self):
        #sobel kernels
        kernel_x = np.array(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], np.float32
        )

        kernel_y = np.array(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]], np.float32
        )
        return kernel_x, kernel_y

    def scale(val):
        #for better visualization
        # (val - min) / (max - min) *255
        return (val - val.min()) / (val.max() - val.min()) * 255

    def calculate_gradient(self):
        kernel_x, kernel_y = self.kernels_for_gradient()
        #correlation from scipy to speed up things a little bit
        G_X = correlate2d(self.output_image_viewer.current_image.modified_image, kernel_x)
        G_Y = correlate2d(self.output_image_viewer.current_image.modified_image, kernel_y)
        # obtaining mag and theta
        total_change = self.scale(np.hypot(G_X, G_Y))
        theta = np.arctan2(G_X, G_Y)
        return total_change, theta

    def non_max_suppression(self, gradient, theta):
        image_height, image_width = self.output_image_viewer.current_image.modified_image.shape
        resultant_img = np.zeros((image_height, image_width), dtype=np.int32)
        # for the angle, max -> 180, min -> -180
        angle = theta * 180.0 / np.pi
        angle[angle < 0] += 180

        for i in range(1, image_height - 1):
            for j in range(1, image_width - 1):
                q = 255
                r = 255

                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    r = gradient[i, j - 1]
                    q = gradient[i, j + 1]

                elif 22.5 <= angle[i, j] < 67.5:
                    r = gradient[i - 1, j + 1]
                    q = gradient[i + 1, j - 1]

                elif 67.5 <= angle[i, j] < 112.5:
                    r = gradient[i - 1, j]
                    q = gradient[i + 1, j]

                elif 112.5 <= angle[i, j] < 157.5:
                    r = gradient[i + 1, j + 1]
                    q = gradient[i - 1, j - 1]

                if (gradient[i, j] >= q) and (gradient[i, j] >= r):
                    resultant_img[i, j] = gradient[i, j]
                else:
                    resultant_img[i, j] = 0
        return resultant_img

    def double_thresholding(self, resultant_img, low_thresh, high_thresh):
        image_height, image_width = resultant_img.shape
        thresholded_img = np.zeros((image_height, image_width), dtype=np.int32)
        for i in range(0, image_height):
            for j in range(0, image_width):
                # lower than low threshold
                if resultant_img[i, j] < low_thresh:
                    thresholded_img[i, j] = 0

                # between thresholds
                elif resultant_img[i, j] >= low_thresh and resultant_img[i, j] < high_thresh:
                    thresholded_img[i, j] = 128

                # higher than high threshold
                else:
                    thresholded_img[i, j] = 255

    def _double_thresholding(g_suppressed, low_threshold, high_threshold):
        g_thresholded = np.zeros(g_suppressed.shape)
        for i in range(0, g_suppressed.shape[0]):  # loop over pixels
            for j in range(0, g_suppressed.shape[1]):
                if g_suppressed[i, j] < low_threshold:  # lower than low threshold
                    g_thresholded[i, j] = 0
                elif g_suppressed[i, j] >= low_threshold and g_suppressed[i, j] < high_threshold:  # between thresholds
                    g_thresholded[i, j] = 128
                else:  # higher than high threshold
                    g_thresholded[i, j] = 255
        return g_thresholded

    # def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    #
    #     highThreshold = img.max() * highThresholdRatio;
    #     lowThreshold = highThreshold * lowThresholdRatio;
    #
    #     M, N = img.shape
    #     res = np.zeros((M, N), dtype=np.int32)
    #
    #     weak = 25
    #     strong = 255
    #
    #     strong_i, strong_j = np.where(img >= highThreshold)
    #     weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    #
    #     res[strong_i, strong_j] = strong
    #     res[weak_i, weak_j] = weak
    #
    #     return res

    def kernel_restrictions(self, kernel_size):
        if kernel_size <3 :
            raise ValueError("kernel size must be >= 3")
        if kernel_size %2 ==0:
            raise ValueError("kernel size must be odd")
        if (kernel_size * kernel_size) < len(self.output_image_viewer):
            raise ValueError("pick a smaller kernel size")



