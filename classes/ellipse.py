import matplotlib.pyplot as plt
# from .shape import Shape
import numpy as np
class Ellipse():
    def __init__(self, center_x, center_y, a, b, theta, num_of_points = 500):
        self.a = a
        self.b = b
        self.center_x = center_x
        self.center_y = center_y
        self.theta = theta
        self.num_of_points = num_of_points
        self.__shape_list = []
        self.__fill_shape_list()
    
    def __fill_shape_list(self):
        t = np.linspace(0,2*np.pi,self.num_of_points)
        x_values = self.center_x + self.a*np.cos(t)*np.cos(self.theta) - self.b*np.sin(t)*np.sin(self.theta)
        y_values = self.center_y + self.a*np.cos(t)*np.sin(self.theta) + self.b*np.sin(t)*np.cos(self.theta)
        self.__shape_list = [x_values, y_values]
        # plt.plot(x_values, y_values, label="Ellipse")
        # # plt.scatter([x_c], [y_c], color="red", label="Center")
        # plt.axis("equal")
        # plt.legend()
        # plt.show()
        
    @property
    def shape_list(self):
        return self.__shape_list