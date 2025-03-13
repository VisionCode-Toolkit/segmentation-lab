import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt5.QtGui import QPainterPath, QColor, QBrush
from PyQt5.QtWidgets import QGraphicsPathItem

from PyQt5.QtGui import QPolygonF, QColor
from PyQt5.QtWidgets import QGraphicsPolygonItem
from PyQt5.QtCore import QPointF
class IntializeContour():
    def __init__(self, output_Image_Viewer):
        # super.__init__()
        self.image_view = output_Image_Viewer
        self.contour_points = []
        self.contour_plot = pg.PlotDataItem([], [], pen={'color': 'b', 'width': 2})
        self.image_view.getView().addItem(self.contour_plot)

        self.image_view.getView().scene().sigMouseClicked.connect(self.add_point)
        self.image_view.getView().scene().sigMouseMoved.connect(self.update_temp_contour)

        # closing contour with a right-click and we will after that enclose it all
        self.image_view.getView().scene().sigMouseClicked.connect(self.check_close_contour)
        self.drawing = True
        self.check_close_contour = False

    def add_point(self, event):
        if self.drawing:
            pos = event.scenePos()
            if self.image_view.getView().sceneBoundingRect().contains(pos):
                mouse_point = self.image_view.getView().mapSceneToView(pos)
                self.contour_points.append((mouse_point.x(), mouse_point.y()))
                self.update_contour()

    def update_contour(self):
        # update the ploting of contour after adding an data point
        if self.drawing:
            if self.contour_points:
                x, y = zip(*self.contour_points)
                self.contour_plot.setData(x, y)


    def update_temp_contour(self, pos):
        # this update only the drawing not the data points
        if self.contour_points and self.drawing:
            last_x, last_y = self.contour_points[-1]
            mouse_point = self.image_view.getView().mapSceneToView(pos)
            x = list(zip(*self.contour_points))[0] + (mouse_point.x(),)
            y = list(zip(*self.contour_points))[1] + (mouse_point.y(),)
            self.contour_plot.setData(x, y)
    def check_close_contour(self, event):
        if event.button() == QtCore.Qt.RightButton and len(self.contour_points) > 2:
            first_x, first_y = self.contour_points[0]
            last_x, last_y = self.contour_points[-1]
            distance = np.hypot(first_x - last_x, first_y - last_y)
            if distance < 10:  # Close the contour if near first point
                self.contour_points.append(self.contour_points[0])
                self.update_contour()
                self.drawing = False
                self.check_close_contour= True
                self.fill_contour()

    def fill_contour(self):
        if len(self.contour_points) > 2 and self.check_close_contour == True:
            path = QPainterPath()
            path.moveTo(*self.contour_points[0])

            for x, y in self.contour_points[1:]:
                path.lineTo(x, y)

            path.closeSubpath()

            fill_item = QGraphicsPathItem(path)
            fill_item.setBrush(QBrush(QColor(0, 0, 255, 100)))  # Blue with transparency


            self.image_view.getView().addItem(fill_item)  # Add to scene

    def clear_contour(self):

        self.contour_plot.setData([], [])
        scene = self.image_view.getView().scene()
        for item in scene.items():
            if isinstance(item, (QGraphicsPathItem, QGraphicsPolygonItem)):
                scene.removeItem(item)

    def update_contour(self):
        if self.check_close_contour == True:
            self.clear_contour()
            self.fill_contour()
    def update_browse_setup(self):
        self.contour_points = []
        self.clear_contour()
        self.check_close_contour = False