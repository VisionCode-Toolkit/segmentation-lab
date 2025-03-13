from classes.imageViewer import ImageViewer
from classes.intializeContour import IntializeContour


class Controller():
    def __init__(self, input_image_viewer:ImageViewer, output_image_viewer:ImageViewer, contour: IntializeContour):
        self.input_image_viewer = input_image_viewer
        self.output_image_viewer = output_image_viewer
        self.contour = contour
        
    def update(self):
        self.input_image_viewer.update_plot()
        self.output_image_viewer.update_plot()
        self.contour.update_contour()