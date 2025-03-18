from scipy.ndimage import maximum_filter
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from classes.line import Line
from classes.circle import Circle
from classes.ellipse import Ellipse
from skimage.measure import ransac, EllipseModel

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
        max_filtered = maximum_filter(accumulator, size=10, mode='constant')
        detected_peaks = (accumulator == max_filtered) & (accumulator > accumulator_threshold)
        peak_indices = np.argwhere(detected_peaks)
        for y_idx, x_idx, r_idx in peak_indices:
            x_center = x_idx * scale_factor
            y_center = y_idx * scale_factor
            radius = radius_range[r_idx]
            self.__output_viewer.current_image.shapes_list.append(Circle(x_center, y_center, radius))
            circles.append([x_center, y_center, radius])
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
        suppressed_accumulator = maximum_filter(accumulator, size = 20, mode='constant')
        line_indices = np.argwhere((accumulator == suppressed_accumulator) & (accumulator > accumulator_threshold))
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
    
    def detect_ellipse(self, num_of_iterations = 90000*4, seed = 60, min_votes = 2): #here we will implement the randomized hough transform 
        self.__output_viewer.current_image.transfer_to_gray_scale()
        image = self.__output_viewer.current_image.modified_image
        height, width = image.shape
        edges = cv2.Canny(image, 50,150)
        y_indices, x_indices = np.where(edges > 0)
        edge_points = np.column_stack((x_indices, y_indices))
        if len(edge_points) < 5:
            return
        # num_of_iterations = min(num_of_iterations, len(edge_points) *5)
        max_axis = max(width, height) // 2
        min_axis = 10
        accumulator = {}
        np.random.seed(seed)
        for _ in range(num_of_iterations):
            points = np.random.choice(len(edge_points), 5, replace=False)
            sample_points = edge_points[points]
            try:
                ellipse = cv2.fitEllipse(sample_points)
                (x_center, y_center), (width_e, height_e), angle = ellipse
                if width_e < min_axis or width_e > max_axis or height < min_axis or height_e > max_axis:
                    continue
                a = width_e / 2
                b = height_e / 2
                x_center_bin = int(x_center)
                y_center_bin = int(y_center)
                a_bin = int(a)
                b_bin = int(b)
                angle_bin = int(angle/5)*5
                parameters = (x_center_bin, y_center_bin, a_bin, b_bin, angle_bin)
                if parameters in accumulator:
                    accumulator[parameters] += 1
                else:
                    accumulator[parameters] = 1
            except Exception as e:
                continue
        ellipses = []
        self.__output_viewer.current_image.shapes_list.clear()
        # accumulator = self.non_max_on_dict(accumulator, kernel_size=10)
        sorted_ellipses = sorted(accumulator.items(), key = lambda x:x[1], reverse=True)
        for parameters, votes in sorted_ellipses:
            if votes < min_votes:
                break
            x_center, y_center, a, b, angle = parameters
            ellipse_element = Ellipse(x_center, y_center, a, b, np.radians(angle))
            self.__output_viewer.current_image.shapes_list.append(ellipse_element)
            ellipses.append(parameters)
            
    def non_max_on_dict(self, dictionary:dict, kernel_size):
        key_set = set()
        for key, item in dictionary.items():
            for x_c in range(kernel_size):
                for y_c in range(kernel_size):
                    for a in range(kernel_size):
                        for b in range(kernel_size):
                            for theta in range(kernel_size):
                                new_key = (key[0] + x_c,key[1] + y_c, key[2]+a, key[3]+b, key[4]+theta)
                                if new_key in dictionary:
                                    if dictionary[new_key] > dictionary[key]:
                                        key_set.add(key)
        for key in key_set:
            dictionary.pop(key, None)
        return dictionary
            
    # def fit_ellipse(points, max_trials = 100, residual_threshold = 2.0):
    #     model = EllipseModel()
    #     inliers = None
    #     try:
    #         model.estimate(points)
    #         inliers = model.predict_xy(points)
    #     except:
    #         pass
    #     x_c, y_c, a, b, theta = model.params
    #     return (x_c, y_c), (a, b), np.rad2deg(theta)
    
    def fit_ellipse(self, points):
        if len(points) < 5:
            raise ValueError("At least 5 points are required to fit an ellipse.")
        x, y = points[:, 0], points[:, 1]
        D = np.column_stack((x**2, x * y, y**2, x, y, np.ones_like(x)))
        _, _, V = np.linalg.svd(D)
        A, B, C, D, E, F = V[-1, :]
        x0 = (C * D - B * E) / (B**2 - A * C)
        y0 = (A * E - B * D) / (B**2 - A * C)
        num = 2 * (A * E**2 + C * D**2 + F * B**2 - 2 * B * D * E - A * C * F)
        denom1 = (B**2 - A * C) * ((C - A) + np.sqrt((A - C)**2 + 4 * B**2))
        denom2 = (B**2 - A * C) * ((A - C) + np.sqrt((A - C)**2 + 4 * B**2))
        major_axis = np.sqrt(num / denom1)
        minor_axis = np.sqrt(num / denom2)
        theta = 0.5 * np.arctan2(2 * B, A - C)
        return (x0, y0), (major_axis, minor_axis) , np.rad2deg(theta)