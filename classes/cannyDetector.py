import cv2
import numpy as np
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

        self.filter.apply_filters(kernel_size, sigma)






    def kernel_restrictions(self, kernel_size):
        if kernel_size <3 :
            raise ValueError("kernel size must be >= 3")
        if kernel_size %2 ==0:
            raise ValueError("kernel size must be odd")
        if (kernel_size * kernel_size) < len(self.output_image_viewer):
            raise ValueError("pick a smaller kernel size")



