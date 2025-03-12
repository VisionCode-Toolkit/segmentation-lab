import cv2
import numpy as np
from scipy.signal import correlate2d

from classes.filter import Filters



class Canny_detector():
    def __init__(self, output_image_viewer):
        self.output_image_viewer = output_image_viewer
        self.filter = Filters(self.output_image_viewer)




    def apply_canny_detector(self, kernel_size, sigma):
        # canny detector can be done in 5 steps
        # make sure the img is gray
        if len(self.output_image_viewer.current_image.modified_image.shape) == 3:
            self.output_image_viewer.current_image.transfer_to_gray_scale()

        # filter the img
        self.output_image_viewer.current_image.modified_image = self.filter.apply_filters(sigma, kernel_size)
        # obtain total change (gradient) and theta
        total_gradient, theta = self.calculate_gradient()

    def kernels_for_gradient(self):
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

    def scale(x):
        return (x - x.min()) / (x.max() - x.min()) * 255

    def calculate_gradient(self):
        kernel_x, kernel_y = self.kernels_for_gradient()
        G_X = correlate2d(self.output_image_viewer.current_image.modified_image, kernel_x)
        G_Y = correlate2d(self.output_image_viewer.current_image.modified_image, kernel_y)
        total_change = self.scale(np.hypot(G_X, G_Y))
        theta = np.arctan2(G_X, G_Y)
        return total_change, theta

    def non_max_suppression(self):
        pass

    def double_thresholding(self, low_thresh, high_thresh):
        pass

    def kernel_restrictions(self, kernel_size):
        if kernel_size <3 :
            raise ValueError("kernel size must be >= 3")
        if kernel_size %2 ==0:
            raise ValueError("kernel size must be odd")
        if (kernel_size * kernel_size) < len(self.output_image_viewer):
            raise ValueError("pick a smaller kernel size")



