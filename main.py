
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFrame, QVBoxLayout, QSlider, QComboBox, QPushButton, \
    QStackedWidget, QWidget, QFileDialog, QRadioButton, QDialog, QLineEdit, QHBoxLayout
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt
from helper_functions.compile_qrc import compile_qrc
from icons_setup.compiledIcons import *
import cv2
from classes.image import Image
from classes.intializeContour import IntializeContour
from classes.imageViewer import ImageViewer
from enums.viewerType import ViewerType
from classes.controller import Controller
from classes.cannyDetector import Canny_detector

from enums.modes import Modes
from classes.snake import ActiveContour
from classes.cannyDetector import Canny_detector

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('main.ui', self)
        self.main_page_browse_button = self.findChild(QPushButton, "browse")
        self.main_page_browse_button.clicked.connect(self.browse_image)

        self.output_image_viewer_layout = self.findChild(QVBoxLayout, "output_layout")
        self.output_image_viewer = ImageViewer()
        self.output_image_viewer_layout.addWidget(self.output_image_viewer)
        self.output_image_viewer.viewer_type = ViewerType.OUTPUT

        self.intialize_snake_model = IntializeContour(self.output_image_viewer)

        self.input_image_viewer_layout = self.findChild(QVBoxLayout, "input_layout")
        self.input_image_viewer = ImageViewer()
        self.input_image_viewer_layout.addWidget(self.input_image_viewer)
        self.input_image_viewer.viewer_type = ViewerType.INPUT

        self.active_contour_model = ActiveContour()

        self.controller = Controller(self.input_image_viewer, self.output_image_viewer, self.intialize_snake_model)

        self.modes_stacked_widget = self.findChild(QStackedWidget, "stackedWidget")
        self.mode_combobox = self.findChild(QComboBox, "mode_combobox")
        self.mode_combobox.currentIndexChanged.connect(self.on_choose_mode_value_changed)

        self.reset_button = self.findChild(QPushButton, "")

        self.kernel_size_text = self.findChild(QLineEdit, "canny_kernel_size")
        self.sigma_text = self.findChild(QLineEdit, "canny_sigma")
        self.low_thresh_text = self.findChild(QLineEdit, "canny_low_threshold")
        self.high_thresh_text = self.findChild(QLineEdit, "canny_high_threshold")

        self.apply_canny_button = self.findChild(QPushButton, "canny_apply_button")
        self.apply_canny_button.clicked.connect(self.apply_canny)
        self.canny_detector = Canny_detector(self.output_image_viewer)


    def browse_image(self):
        print("pushed")
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Image Files (*.jpeg *.jpg *.png *.bmp *.gif);;All Files (*)')

        if file_path:
            if file_path.endswith('.jpeg') or file_path.endswith('.jpg'):
                temp_image = cv2.imread(file_path)
                image = Image(temp_image)

                self.input_image_viewer.current_image = image
                self.output_image_viewer.current_image = image

                self.intialize_snake_model.update_browse_setup()
                text = self.mode_combobox.currentText()
                if text == Modes.SNAKE.value:
                    self.intialize_snake_model.drawing = True
                else:
                    self.intialize_snake_model.drawing = False
# passing by reference
                self.active_contour_model.set_image(self.output_image_viewer.current_image)
                self.active_contour_model.set_contour(self.intialize_snake_model.contour_points)

                # update
                self.controller.update()

    def on_choose_mode_value_changed(self):
        text = self.mode_combobox.currentText()
        print(text)
        # return
        if text == Modes.CANNY.value:
            page_index = self.modes_stacked_widget.indexOf(self.findChild(QWidget, "canny_page"))
        elif text == Modes.SNAKE.value:
            page_index = self.modes_stacked_widget.indexOf(self.findChild(QWidget, "snake_page"))

        elif text == Modes.LINE.value:
            page_index = self.modes_stacked_widget.indexOf(self.findChild(QWidget, "Hough_line_page"))
        elif text == Modes.CIRCLE.value:
            page_index = self.modes_stacked_widget.indexOf(self.findChild(QWidget, "hough_circle_page"))
        elif text == Modes.ELLIPSE.value:
            page_index = self.modes_stacked_widget.indexOf(self.findChild(QWidget, "Hough_elipse_page"))
        else:
            page_index = self.modes_stacked_widget.indexOf(self.findChild(QWidget, "main_page"))
            self.intialize_snake_model.drawing = False
        if text == Modes.SNAKE.value:
            self.intialize_snake_model.drawing = True
        else:
            print("iam in the false")
            self.intialize_snake_model.drawing = False

        if page_index != -1:
            self.modes_stacked_widget.setCurrentIndex(page_index)

    def apply_canny(self):
        kernel_size = int(self.kernel_size_text.text())
        sigma = float(self.sigma_text.text())
        low_thresh = float(self.low_thresh_text.text())
        high_thresh = float(self.low_thresh_text.text())
        print(kernel_size, sigma, low_thresh, high_thresh)

        self.canny_detector.apply_canny_detector(kernel_size, sigma, low_thresh, high_thresh)
        self.controller.update()


    
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())