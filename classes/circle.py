from .shape import Shape
import numpy as np
class Circle(Shape):
    def __init__(self, center_x, center_y, radius, num_of_points = 500):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.num_of_points = num_of_points
        self.__shape_list = []
        self.__fill_shape_list()
    
    def __fill_shape_list(self):
        theta = np.linspace(0,2*np.pi, self.num_of_points)
        x_values = self.radius * np.sin(theta) + self.center_x
        y_values = self.radius * np.cos(theta) + self.center_y
        self.__shape_list = [x_values, y_values]
        
    @property
    def shape_list(self):
        return self.__shape_list
    
    