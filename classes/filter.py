import math
import cv2
import numpy as np


class Filters():
    def __init__(self, output_image_viewer):
        # redundant for now
        self.output_image_viewer = output_image_viewer
    def apply_filters(self, kernel_size, sigma):
        # feel free to add filters needed here (ASK ME FIRST)
        print("gwa apply filter")
        filtered_img = self.apply_gaussian_filter(kernel_size, sigma)
        return filtered_img

    def create_gaussian_kernel(self, kernel_size, sigma):
        gaussian_kernel = []
        total_sum = 0
        for i in range(int(-(kernel_size - 1) / 2), int((kernel_size - 1) / 2 + 1)):
            filter_row = []
            for j in range(int(-(kernel_size - 1) / 2), int((kernel_size - 1) / 2 + 1)):
                G = math.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
                filter_row.append(G)
                total_sum += G
            gaussian_kernel.append(filter_row)

        gaussian_kernel = np.array(gaussian_kernel) / total_sum
        return gaussian_kernel

    def apply_gaussian_filter(self, sigma, kernel_size=11):
        image_height, image_width = self.output_image_viewer.current_image.modified_image.shape
        filtered_img = np.zeros_like(self.output_image_viewer.current_image.modified_image, dtype=np.float32)

        # creating the gaussian kernel
        kernel = self.create_gaussian_kernel(kernel_size, sigma)
        # creating padding (fake pixels to handle edges)
        pad_size = kernel_size // 2
        padded_image = np.pad(self.output_image_viewer.current_image.modified_image
                              ,((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')

        for i in range(pad_size, image_height + pad_size):
            for j in range(pad_size, image_width + pad_size):
                # we apply the process mat by mat
                region = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
                # apply the Gaussian kernel
                filtered_img[i - pad_size, j - pad_size] = np.sum(region * kernel)

        filtered_img = filtered_img.astype(np.uint8)
        return filtered_img



