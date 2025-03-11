from classes.imageViewer import ImageViewer


class Controller():
    def __init__(self, input_image_viewer:ImageViewer, output_image_viewer:ImageViewer):
        self.input_image_viewer = input_image_viewer
        self.output_image_viewer = output_image_viewer
        
    def update(self):
        self.input_image_viewer.update_plot()
        self.output_image_viewer.update_plot()