
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFrame, QVBoxLayout, QSlider, QComboBox, QPushButton, \
    QStackedWidget, QWidget, QFileDialog, QRadioButton, QDialog, QLineEdit, QHBoxLayout
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QTimer
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
from classes.hough import Hough



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

        self.reset_button = self.findChild(QPushButton, "pushButton_2")
        self.reset_button.clicked.connect(self.on_reset_button_clicked)
        
        self.line_apply_button = self.findChild(QPushButton, "line_apply_button")
        self.line_apply_button.clicked.connect(self.on_detect_line_clicked)
        
        self.circle_apply_button = self.findChild(QPushButton, "circle_apply_button")
        self.circle_apply_button.clicked.connect(self.on_detect_circle_clicked)
        
        self.ellipse_apply_button = self.findChild(QPushButton, "ellipse_apply_button")
        self.ellipse_apply_button.clicked.connect(self.on_detect_ellipse_clicked)

        self.kernel_size_text = self.findChild(QLineEdit, "canny_kernel_size")
        self.sigma_text = self.findChild(QLineEdit, "canny_sigma")
        self.low_thresh_text = self.findChild(QLineEdit, "canny_low_threshold")
        self.high_thresh_text = self.findChild(QLineEdit, "canny_high_threshold")

        self.apply_canny_button = self.findChild(QPushButton, "canny_apply_button")
        self.apply_canny_button.clicked.connect(self.apply_canny)
        self.canny_detector = Canny_detector(self.output_image_viewer)
        self.hough = Hough(self.output_image_viewer)

        self.statistics_widget = self.findChild(QWidget, "statistics_widget")
        self.statistics_widget.hide()

        # apply snake Model
        self.apply_icon = QIcon("icons_setup/icons/play-button-arrowhead.png")
        self.pause_icon = QIcon("icons_setup/icons/pause.png")
        self.is_apply_icon = True

        self.apply_snake_model_button = self.findChild(QPushButton, "snake_apply_button")
        self.apply_snake_model_button.clicked.connect(self.apply_snake)


        # set snake model _ parameters
        self.set_aplha_model_label = self.findChild(QLineEdit, "snake_alpha")
        self.set_aplha_model_label.returnPressed.connect(lambda: self.apply_snake_parameters(self.set_aplha_model_label.text(), "alpha"))

        self.set_gamma_model_label = self.findChild(QLineEdit, "snake_gamma")
        self.set_gamma_model_label.returnPressed.connect(lambda: self.apply_snake_parameters(self.set_gamma_model_label.text(), "gamma"))

        self.set_beta_model_label = self.findChild(QLineEdit, "snake_beta")
        self.set_beta_model_label.returnPressed.connect(lambda: self.apply_snake_parameters(self.set_beta_model_label.text(), "beta"))

        self.set_num_of_iterations = self.findChild(QLineEdit, "snake_no_of_iterations")
        self.set_num_of_iterations.returnPressed.connect(lambda: self.apply_snake_parameters(self.set_num_of_iterations.text(), "iteration"))

        self.chain_code = self.findChild(QLabel, "chain_code")
        self.perimeter_of_contour = self.findChild(QLabel, "perimeter_of_contour")
        self.area_of_contour = self.findChild(QLabel, "area_of_contour")

        # apply timer to manage pause event :
        self.timer = QTimer()
        self.timer.timeout.connect(self.evolvong_With_drawing)

    def browse_image(self):
        print("pushed")
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Image Files (*.jpeg *.jpg *.png *.bmp *.gif);;All Files (*)')

        if file_path:
            if file_path.endswith('.jpeg') or file_path.endswith('.jpg'):
                temp_image = cv2.imread(file_path)
                image = Image(temp_image)
                print(image.original_image.shape)

                self.input_image_viewer.current_image = image
                self.output_image_viewer.current_image = image

                self.intialize_snake_model.update_setup()
                self.check_enable_drawing()


# passing by reference
                self.active_contour_model.set_image(self.output_image_viewer.current_image.original_image)
                self.active_contour_model.set_contour(self.intialize_snake_model.contour_points)

                # update
                self.controller.update()


    def check_enable_drawing(self):
        text = self.mode_combobox.currentText()
        if text == Modes.SNAKE.value:
            self.intialize_snake_model.drawing = True
            self.active_contour_model.current_iteration = 0
        else:
            self.intialize_snake_model.drawing = False


    def on_choose_mode_value_changed(self):
        text = self.mode_combobox.currentText()
        print(text)
        # return
        if text == Modes.CANNY.value:
            page_index = self.modes_stacked_widget.indexOf(self.findChild(QWidget, "canny_page"))
        elif text == Modes.SNAKE.value:
            page_index = self.modes_stacked_widget.indexOf(self.findChild(QWidget, "snake_page"))
            self.statistics_widget.show()

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
            
    def on_detect_circle_clicked(self):
        self.hough.detect_circles()
        self.controller.update()
    
    def on_detect_line_clicked(self):
        accumulator_threshold = int(self.findChild(QLineEdit, "line_no_of_lines").text())
        low_threshold = int(self.findChild(QLineEdit, "line_low_threshold").text())
        high_threshold = int(self.findChild(QLineEdit, "line_high_threshold").text())
        self.hough.detect_lines(accumulator_threshold, low_threshold, high_threshold)
        self.controller.update()
    
    def on_detect_ellipse_clicked(self):
        self.hough.detect_ellipse()
        self.controller.update()

    def apply_canny(self):
        kernel_size = int(self.kernel_size_text.text())
        sigma = float(self.sigma_text.text())
        low_thresh = float(self.low_thresh_text.text())
        high_thresh = float(self.low_thresh_text.text())
        print(kernel_size, sigma, low_thresh, high_thresh)
        self.canny_detector.apply_canny_detector(kernel_size, sigma, low_thresh, high_thresh)
        self.controller.update()
        
    def on_reset_button_clicked(self):
        self.output_image_viewer.current_image.reset()
        self.intialize_snake_model.update_setup()
        self.check_enable_drawing()
        self.controller.update()


    def evolvong_With_drawing(self):
        self.active_contour_model.evolve_step()
        self.intialize_snake_model.contour_points = list(self.active_contour_model.contour)
        self.controller.update()

    def show_statistics(self):
        chain_code = self.active_contour_model.compute_chain_code()
        self.chain_code.setText(chain_code)

        perimeter = self.active_contour_model.compute_perimeter()
        self.perimeter_of_contour.setText(perimeter)

        area = self.active_contour_model.compute_area()
        self.area_of_contour.setText(area)

    def apply_snake(self):
        print("apply snake")
        if self.is_apply_icon :
            self.apply_snake_model_button.setIcon(self.pause_icon)
            self.apply_snake_model_button.setText("Pause")
            self.is_apply_icon = False
            self.apply_snake_model_button.repaint()
            self.active_contour_model.set_contour(self.intialize_snake_model.contour_points)
            self.active_contour_model.flag_continue = True

            self.active_contour_model.flag_continue = True
            self.timer.start(100)


        else :
            self.active_contour_model.flag_continue = False
            self.apply_snake_model_button.setIcon(self.apply_icon)
            self.apply_snake_model_button.setText("Play")
            self.is_apply_icon = True
            self.apply_snake_model_button.repaint()
            self.timer.stop()

        self.intialize_snake_model.contour_points = list(self.active_contour_model.contour)
        self.controller.update()

    def apply_snake_parameters(self, value, parameter_Type):
        if parameter_Type == "iteration" :
            self.active_contour_model.max_iterations = int(value)
        elif parameter_Type == "alpha" :
            self.active_contour_model.alpha = float(value)
        elif parameter_Type == "gamma" :
            self.active_contour_model.gamma = float(value)
        elif parameter_Type == "beta" :
            self.active_contour_model.beta = float(value)
        else :
            self.active_contour_model.window_size = int(value)
        self.controller.update()




    
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())