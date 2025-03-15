import cv2
import numpy as np
import matplotlib.pyplot as plt 
from classes.line import Line
from classes.circle import Circle
class Hough():
    def __init__(self, output_viewer):
        self.__output_viewer = output_viewer
    
    
    def detect_circles(self, accumulator_threshold=50):
        self.__output_viewer.current_image.transfer_to_gray_scale()
        image = self.__output_viewer.current_image.modified_image
        scale = 1
        small_image = cv2.resize(image, None, fx=scale, fy=scale)
        height, width = small_image.shape
        scale_factor = 2  
        acc_height, acc_width = height//scale_factor, width//scale_factor
        min_radius, max_radius = 10, min(width, height)//2
        radius_range = np.arange(min_radius, max_radius, 2)  # Step by 2 for speed
        accumulator = np.zeros((acc_height, acc_width, len(radius_range)))
        edges = cv2.Canny(small_image, 50, 150)
        y_indices, x_indices = np.where(edges > 0)
        
        test = 0

        for i in range(len(x_indices)):
            x, y = x_indices[i], y_indices[i]
            for r_idx, radius in enumerate(radius_range):
                for angle in range(0, 360, 10):  # Step by 10 degrees for speed
                    a = x - radius * np.cos(angle * np.pi / 180)
                    b = y - radius * np.sin(angle * np.pi / 180)
                    test+=1
                    if 0 <= a < width and 0 <= b < height:
                        accumulator[int(b//scale_factor), int(a//scale_factor), r_idx] += 1
        circles = []
        threshold = accumulator_threshold
        self.__output_viewer.current_image.shapes_list.clear()
        for r_idx, radius in enumerate(radius_range):
            from scipy.ndimage import maximum_filter
            local_max = maximum_filter(accumulator[:,:,r_idx], size=5)
            detected_peaks = (accumulator[:,:,r_idx] == local_max) & (accumulator[:,:,r_idx] > threshold)
            y_peaks, x_peaks = np.where(detected_peaks)
            for i in range(len(x_peaks)):
                x_center = x_peaks[i] * scale_factor
                y_center = y_peaks[i] * scale_factor
                self.__output_viewer.current_image.shapes_list.append(Circle(x_center/scale, y_center/scale, radius/scale))
                circles.append([x_center/scale, y_center/scale, radius/scale])
        print(circles)
        return circles
        
    
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
                rho = int(x * np.cos(theta) + y * np.sin(theta))
                rho_idx = np.argmin(np.abs(rho_range - rho)) # Find closest rho value index
                accumulator[rho_idx, theta_idx] += 1
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