import pyqtgraph as pg
from enums.viewerType import ViewerType
import cv2


class ImageViewer(pg.ImageView):
    def __init__(self):
        super().__init__()
        self.getView().setBackgroundColor("#edf6f9")
        self.ui.histogram.hide()
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.getView().setAspectLocked(False)
        self.current_image = None

    def update_plot(self):
        if self.current_image is not None:
            self.clear()
            if self.viewer_type == ViewerType.INPUT:
                self.setImage(cv2.transpose(self.current_image.original_image))
            elif self.viewer_type == ViewerType.OUTPUT:
                self.setImage(cv2.transpose(self.current_image.modified_image))
                ## important note: we need to plot the shapes or the contours on the top of the modified image,
                ## so you need to plot it after the plotting of the image itself but do not put the contours in the list of the modified image
                ## you have a separated list for yourself