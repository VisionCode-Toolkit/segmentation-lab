from .shape import Shape
import numpy as np
class Line(Shape):
    def __init__(self, x_start, x_end, m, b, num_of_points = 500):
        self.x_start = x_start
        self.x_end = x_end
        self.m = m
        self.b = b
        self.num_of_points = num_of_points
        self.__shape_list = []
        self.__fill_shape_list()
    
    def __fill_shape_list(self):
        x_values = np.linspace(self.x_start, self.x_end, self.num_of_points)
        y_values = self.m * x_values + self.b
        self.shape_list = [x_values, y_values]
        
    @property
    def shape_list(self):
        return self.__shape_list
    
    