import cv2
import numpy as np
import matplotlib.pyplot as plt 
from classes.line import Line
class Hough():
    def __init__(self, output_viewer):
        self.__output_viewer = output_viewer
    
    
    def detect_circles(self):
        pass
    
    def detect_lines(self, accumulator_threshold=40, low_threshold = 50, high_threshold = 250 ):
        self.__output_viewer.current_image.transfer_to_gray_scale()
        
        # Make the grid
        image = self.__output_viewer.current_image.modified_image
        height, width = image.shape
        theta_range = np.deg2rad(np.linspace(-90, 90, 180))  # Convert to radians
        rho_max = int(np.sqrt(height**2 + width**2))
        rho_range = np.linspace(-rho_max, rho_max, 2*rho_max)
        accumulator = np.zeros((len(rho_range), len(theta_range)), dtype=np.int32)
        # Make edge detection
        image = cv2.Canny(self.__output_viewer.current_image.modified_image, low_threshold, high_threshold)
        # For each point in the edges, compute all possible rho and theta values
        y_indices, x_indices = np.where(image > 0)  # Get row (y) and column (x) indices of edges
        for i in range(len(y_indices)):
            y = y_indices[i]
            x = x_indices[i]
            for theta_idx, theta in enumerate(theta_range):
                # Calculate rho using proper formula with x and y in the correct positions
                rho = int(x * np.cos(theta) + y * np.sin(theta))
                # Find closest rho value index
                rho_idx = np.argmin(np.abs(rho_range - rho))
                # Increment accumulator
                accumulator[rho_idx, theta_idx] += 1
        # Find lines that exceed threshold
        line_indices = np.argwhere(accumulator > accumulator_threshold)
        # Convert indices back to actual rho and theta values
        lines = []
        self.__output_viewer.current_image.shapes_list.clear()
        for line in line_indices:
            rho_idx, theta_idx = line
            rho = rho_range[rho_idx]
            theta = theta_range[theta_idx]
            line = Line(rho, theta)
            self.__output_viewer.current_image.shapes_list.append(line)
            lines.append([rho, np.rad2deg(theta)])  # Convert theta back to degrees
        
        print(lines)
        return lines  # Return the actual lines
    
    def detect_ellipse(self):
        pass