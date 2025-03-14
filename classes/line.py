# from .shape import Shape
import numpy as np
import matplotlib.pyplot as plt

class Line():
    def __init__(self,rho, theta, t_range = (-0, 500), x_start=0, x_end = 500, num_of_points = 500):
        self.x_start = x_start
        self.x_end = x_end
        self.rho = rho 
        self.theta = theta
        self.t_range = t_range
        self.num_of_points = num_of_points
        self.__shape_list = []
        self.__fill_shape_list()
    
    def __fill_shape_list(self):
        t = np.linspace(self.t_range[0], self.t_range[1], self.num_of_points)
        x_values = self.rho * np.cos(self.theta) + t * (-np.sin(self.theta))
        y_values = self.rho * np.sin(self.theta) + t * np.cos(self.theta)
        self.__shape_list = [x_values, y_values]
        
    @property
    def shape_list(self):
        return self.__shape_list
    
    