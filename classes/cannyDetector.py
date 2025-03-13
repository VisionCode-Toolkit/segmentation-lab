import cv2
import numpy as np
from scipy.signal import correlate2d
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize


from classes.filter import Filters



class Canny_detector():
    def __init__(self, output_image_viewer):
        self.output_image_viewer = output_image_v addiewer
        self.filter = Filters(self.output_image_viewer)

    def apply_canny_detector(self, kernel_size, sigma, low_thresh, high_tresh):
        if self.output_image_viewer.current_image is not None:
            # canny detector can be done in 5 steps
            # make sure the img is gray
            if len(self.output_image_viewer.current_image.modified_image.shape) == 3:
                print("img is not gray")
                self.output_image_viewer.current_image.transfer_to_gray_scale()

            # filter the img
            self.output_image_viewer.current_image.modified_image = self.filter.apply_filters(sigma, kernel_size)
            # obtain total change (gradient) and theta
            total_gradient, theta = self.calculate_gradient()

            # # obtain resultant image from nms
            resultant_img = self.non_max_suppression(total_gradient,theta)

            # thresholding
            thresholded_img = self.double_thresholding(resultant_img, low_thresh, high_tresh)
            final_output = self.apply_hysteresis(thresholded_img)
            # final_output = self.apply_morphological_operations(final_output, kernel_size=3)
            final_output = self.thin_edges(final_output)
            self.output_image_viewer.current_image.modified_image = final_output

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

    @staticmethod
    def scale(val):
        #for better visualization
        # (val - min) / (max - min) *255
        return (val - val.min()) / (val.max() - val.min()) * 255



    def gradient_mag(self, gx, gy):
        mag = np.zeros((gx.shape[0], gy.shape[1]))
        mag = np.sqrt((gx ** 2) + (gy ** 2))
        mag = mag * 100 / mag.max()
        return np.around(mag)



    def calculate_gradient(self):
        kernel_x, kernel_y = self.kernels_for_gradient()
        #correlation from scipy to speed up things a little bit
        # mode = "same" --> to ensure the output has the same size as the input
        # G_X = correlate2d(self.output_image_viewer.current_image.modified_image, kernel_x, mode='same')
        # G_Y = correlate2d(self.output_image_viewer.current_image.modified_image, kernel_y, mode='same')
        G_X = cv2.filter2D(self.output_image_viewer.current_image.modified_image, -1, kernel_x,
                           borderType=cv2.BORDER_CONSTANT)
        G_Y = cv2.filter2D(self.output_image_viewer.current_image.modified_image, -1, kernel_y,
                           borderType=cv2.BORDER_CONSTANT)
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
        for i in range(1, image_height -1):
            for j in range(1, image_width -1):
                # lower than low threshold
                if resultant_img[i, j] < low_thresh:
                    thresholded_img[i, j] = 0

                # between thresholds
                elif resultant_img[i, j] >= low_thresh and resultant_img[i, j] < high_thresh:
                    thresholded_img[i, j] = 128

                # higher than high threshold
                else:
                    thresholded_img[i, j] = 255

        return thresholded_img


    def apply_hysteresis(self, thresholded_img):
        image_height, image_width = thresholded_img.shape
        final_output = np.zeros((image_height, image_width), dtype=np.uint8)
        for i in range(1, image_height - 1):
            for j in range(1, image_width - 1):
                val = thresholded_img[i, j]
                if val == 128:
                    if (thresholded_img[i - 1, j] == 255 or thresholded_img[i + 1, j] == 255 or
                            thresholded_img[i - 1, j - 1] == 255 or thresholded_img[i + 1, j - 1] == 255 or
                            thresholded_img[i - 1, j + 1] == 255 or thresholded_img[i + 1, j + 1] == 255 or
                            thresholded_img[i, j - 1] == 255 or thresholded_img[i, j + 1] == 255):
                        final_output[i, j] = 255
                    else:
                        final_output[i, j] = 0
                elif val == 255:
                    final_output[i, j] = 255
        return final_output

    def apply_morphological_operations(self, edges, kernel_size=3):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)
        return closed_edges

    def thin_edges(self, edges):
        return skeletonize(edges // 255).astype(np.uint8) * 255





    def kernel_restrictions(self, kernel_size):
        if kernel_size <3 :
            raise ValueError("kernel size must be >= 3")
        if kernel_size %2 ==0:
            raise ValueError("kernel size must be odd")
        if (kernel_size * kernel_size) > (self.output_image_viewer.current_image.modified_image.shape[0] *
                                          self.output_image_viewer.current_image.modified_image.shape[1]):
            raise ValueError("pick a smaller kernel size")


# https://github.com/UsamaI000/CannyEdgeDetection-from-scratch-python/blob/master/CannyEdgeDetector.ipynb


    # def apply_hysteresis(self, thresholded_img):
    #     image_height, image_width = thresholded_img.shape
    #     final_output = np.zeros((image_height, image_width), dtype=np.int32)
    #     for i in range(1, image_height - 1):
    #         for j in range(1, image_width - 1):
    #             val = thresholded_img[i, j]
    #             # if a weak edge connected to strong
    #             if val == 128:
    #                 if thresholded_img[i - 1, j] == 255 or thresholded_img[i + 1, j] == 255 or thresholded_img[
    #                     i - 1, j - 1] == 255 or thresholded_img[i + 1, j - 1] == 255 or thresholded_img[
    #                     i - 1, j + 1] == 255 or thresholded_img[i + 1, j + 1] == 255 or thresholded_img[i, j - 1] == 255 or \
    #                         thresholded_img[i, j + 1] == 255:
    #                     # replace weak edge as strong
    #                     final_output[i, j] = 255
    #             elif val == 255:
    #                 # strong edge remains the same
    #                 final_output[i, j] = 255
    #     return final_output

    # def double_thresholding(self, img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
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
    #     # zeros_i, zeros_j = np.where(img < lowThreshold)
    #
    #     weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    #
    #     res[strong_i, strong_j] = strong
    #     res[weak_i, weak_j] = weak
    #
    #     return res


    # def apply_hysteresis(self, img):
    #     weak = 25
    #     strong = 255
    #     M, N = img.shape
    #
    #     for i in range(1, M - 1):
    #         for j in range(1, N - 1):
    #             if (img[i, j] == weak):
    #                 if (
    #                         (img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or
    #                         (img[i + 1, j + 1] == strong) or (img[i, j - 1] == strong) or
    #                         (img[i, j + 1] == strong) or (img[i - 1, j - 1] == strong) or
    #                         (img[i - 1, j] == strong) or (img[i - 1, j + 1] == strong)
    #                 ):
    #                     img[i, j] = strong
    #                 else:
    #                     img[i, j] = 0
    #     return img
