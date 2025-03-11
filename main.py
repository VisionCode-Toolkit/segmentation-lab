
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
from classes.imageViewer import ImageViewer
from enums.viewerType import ViewerType
from classes.controller import Controller
from enums.modes import Modes

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
        
        self.input_image_viewer_layout = self.findChild(QVBoxLayout, "input_layout")
        self.input_image_viewer = ImageViewer()
        self.input_image_viewer_layout.addWidget(self.input_image_viewer)
        self.input_image_viewer.viewer_type = ViewerType.INPUT
        
        self.controller = Controller(self.input_image_viewer, self.output_image_viewer)
        
        self.modes_stacked_widget = self.findChild(QStackedWidget, "stackedWidget")
        self.mode_combobox = self.findChild(QComboBox, "mode_combobox")
        self.mode_combobox.currentIndexChanged.connect(self.on_choose_mode_value_changed)
        
        self.reset_button = self.findChild(QPushButton, "")
        
    def browse_image(self):
        print("pushed")
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Image Files (*.jpeg *.jpg *.png *.bmp *.gif);;All Files (*)')
        
        if file_path:
            if file_path.endswith('.jpeg') or file_path.endswith('.jpg'):
                temp_image = cv2.imread(file_path)
                image = Image(temp_image)
                
                self.input_image_viewer.current_image = image 
                self.output_image_viewer.current_image = image
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
            
            
        if page_index != -1:
            self.modes_stacked_widget.setCurrentIndex(page_index)
    
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())