import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from scipy.signal import correlate2d

# Assuming the Filters class is defined elsewhere
class Filters:
    def __init__(self, output_image_viewer):
        self.output_image_viewer = output_image_viewer

    def apply_filters(self, sigma, kernel_size):
        # Placeholder for filter application
        return cv2.GaussianBlur(self.output_image_viewer.current_image.modified_image, (kernel_size, kernel_size), sigma)

class Canny_detector:
    def __init__(self, output_image_viewer):
        self.output_image_viewer = output_image_viewer
        self.filter = Filters(self.output_image_viewer)

    def apply_canny_detector(self, kernel_size, sigma, low_thresh, high_tresh):
        # Ensure the image is grayscale
        if len(self.output_image_viewer.current_image.modified_image.shape) == 3:
            self.output_image_viewer.current_image.transfer_to_gray_scale()

        # Apply Gaussian blur
        self.output_image_viewer.current_image.modified_image = self.filter.apply_filters(sigma, kernel_size)

        # Calculate gradient
        total_gradient, theta = self.calculate_gradient()

        # Apply non-maximum suppression
        resultant_img = self.non_max_suppression(total_gradient, theta)

        # Apply double thresholding
        thresholded_img = self.double_thresholding(resultant_img, low_thresh, high_tresh)

        # Apply hysteresis
        final_output = self.apply_hysteresis(thresholded_img)
        self.output_image_viewer.current_image.modified_image = final_output

    def kernels_for_gradient(self):
        # Sobel kernels
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
        return kernel_x, kernel_y

    @staticmethod
    def scale(val):
        # Normalize the image to the range [0, 255]
        return (val - val.min()) / (val.max() - val.min()) * 255

    def calculate_gradient(self):
        kernel_x, kernel_y = self.kernels_for_gradient()
        G_X = cv2.filter2D(self.output_image_viewer.current_image.modified_image, -1, kernel_x)
        G_Y = cv2.filter2D(self.output_image_viewer.current_image.modified_image, -1, kernel_y)
        total_change = self.scale(np.hypot(G_X, G_Y))
        theta = np.arctan2(G_Y, G_X)
        return total_change, theta

    def non_max_suppression(self, gradient, theta):
        image_height, image_width = gradient.shape
        resultant_img = np.zeros((image_height, image_width), dtype=np.int32)
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
        for i in range(1, image_height - 1):
            for j in range(1, image_width - 1):
                if resultant_img[i, j] < low_thresh:
                    thresholded_img[i, j] = 0
                elif resultant_img[i, j] >= low_thresh and resultant_img[i, j] < high_thresh:
                    thresholded_img[i, j] = 128
                else:
                    thresholded_img[i, j] = 255
        return thresholded_img

    def apply_hysteresis(self, thresholded_img):
        image_height, image_width = thresholded_img.shape
        final_output = np.zeros((image_height, image_width), dtype=np.int32)
        for i in range(1, image_height - 1):
            for j in range(1, image_width - 1):
                val = thresholded_img[i, j]
                if val == 128:
                    if thresholded_img[i - 1, j] == 255 or thresholded_img[i + 1, j] == 255 or \
                       thresholded_img[i - 1, j - 1] == 255 or thresholded_img[i + 1, j - 1] == 255 or \
                       thresholded_img[i - 1, j + 1] == 255 or thresholded_img[i + 1, j + 1] == 255 or \
                       thresholded_img[i, j - 1] == 255 or thresholded_img[i, j + 1] == 255:
                        final_output[i, j] = 255
                elif val == 255:
                    final_output[i, j] = 255
        return final_output

    def kernel_restrictions(self, kernel_size):
        if kernel_size < 3:
            raise ValueError("kernel size must be >= 3")
        if kernel_size % 2 == 0:
            raise ValueError("kernel size must be odd")
        if (kernel_size * kernel_size) > self.output_image_viewer.current_image.modified_image.shape[0] * self.output_image_viewer.current_image.modified_image.shape[1]:
            raise ValueError("pick a smaller kernel size")

class ImageViewer:
    def __init__(self):
        self.current_image = self.Image()

    class Image:
        def __init__(self):
            self.modified_image = None

        def transfer_to_gray_scale(self):
            if len(self.modified_image.shape) == 3:
                self.modified_image = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2GRAY)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Canny Edge Detector")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setText("Upload an image to apply Canny Edge Detection")
        self.label.setAlignment(Qt.AlignCenter)

        self.btn_upload = QPushButton("Upload Image", self)
        self.btn_upload.clicked.connect(self.upload_image)

        self.btn_apply_canny = QPushButton("Apply Canny", self)
        self.btn_apply_canny.clicked.connect(self.apply_canny)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn_upload)
        layout.addWidget(self.btn_apply_canny)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.image_viewer = ImageViewer()
        self.canny_detector = Canny_detector(self.image_viewer)

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.image_viewer.current_image.modified_image = cv2.imread(file_name)
            self.display_image(self.image_viewer.current_image.modified_image)

    def apply_canny(self):
        if self.image_viewer.current_image.modified_image is not None:
            self.canny_detector.apply_canny_detector(kernel_size=3, sigma=1.0, low_thresh=50, high_tresh=150)
            self.display_image(self.image_viewer.current_image.modified_image)

    def display_image(self, image):
        if len(image.shape) == 2:
            h, w = image.shape
            q_img = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_BGR888)

        self.label.setPixmap(QPixmap.fromImage(q_img))
        self.label.setAlignment(Qt.AlignCenter)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())