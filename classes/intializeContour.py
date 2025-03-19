import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt5.QtGui import QPainterPath, QColor, QBrush
from PyQt5.QtWidgets import QGraphicsPathItem, QColorDialog, QPushButton, QVBoxLayout, QWidget

from PyQt5.QtGui import QPolygonF, QColor
from PyQt5.QtWidgets import QGraphicsPolygonItem
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QPalette, QColor
class IntializeContour():
    def __init__(self, output_Image_Viewer):
        # super.__init__()
        self.image_view = output_Image_Viewer
        self.contour_points = []
        self.contour_color = 'blue'
        self.fill_color = QColor(0, 0, 255, 100)
        self.contour_plot = pg.PlotDataItem([], [], pen={'color': self.contour_color, 'width': 2})
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
                self.update_contour_data()

    def update_contour_data(self):
        # update the ploting of contour after adding an data point
        if self.drawing:
            if self.contour_points:
                x, y = zip(*self.contour_points)
                self.contour_plot.setData(x, y)


    def update_temp_contour(self, pos): # to show if you go there what will happen
        # this update only the drawing not the data points
        if self.contour_points and self.drawing:
            last_x, last_y = self.contour_points[-1]
            mouse_point = self.image_view.getView().mapSceneToView(pos)
            x = list(zip(*self.contour_points))[0] + (mouse_point.x(),)
            y = list(zip(*self.contour_points))[1] + (mouse_point.y(),)
            self.contour_plot.setData(x, y)
    def check_close_contour(self, event): # check if the contour is closed loop
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
                self.update_colors()

    def fill_contour(self): # refill / fill the contour as closed loop
        if len(self.contour_points) > 2 and self.check_close_contour == True:
            path = QPainterPath()
            path.moveTo(*self.contour_points[0])

            for x, y in self.contour_points[1:]:
                path.lineTo(x, y)

            path.closeSubpath()

            fill_item = QGraphicsPathItem(path)
            fill_item.setBrush(QBrush(self.fill_color))  # blue with transparency


            self.image_view.getView().addItem(fill_item)  # Add to scene

    def clear_contour(self): # clear contour plot

        self.image_view.getView().removeItem(self.contour_plot)
        scene = self.image_view.getView().scene()
        for item in scene.items():
            if isinstance(item, (QGraphicsPathItem, QGraphicsPolygonItem)):
                scene.removeItem(item)
        self.contour_plot = pg.PlotDataItem([], [], pen={'color':self.contour_color, 'width': 2})
        self.image_view.getView().addItem(self.contour_plot)



    def update_contour(self): # to replot the contour in the controller updates
        if self.check_close_contour == True:
            self.clear_contour()
            self.fill_contour()
    def update_setup(self): # to clear contour line and draw
        self.contour_points = []
        self.clear_contour()
        self.check_close_contour = False

    def update_colors(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.contour_color = color.name()
            self.contour_plot.setPen(pg.mkPen(self.contour_color, width=2))
            self.fill_color = QColor(color.red(), color.green(), color.blue(), 100)
            self.update_contour()

