import numpy as np
import cv2
import matplotlib.pyplot as plt

class ActiveContour:
    def __init__(self, alpha: float, beta: float, gamma: float, num_iterations: int):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_iterations = num_iterations
        self.image = None  
        self.contour = None 
        self.chain_code = None  

    def upload_image(self, image: np.ndarray): 
        """
        upload or change the input image after the user change it in the main. 
        we will call it in the upload function in the main
        
        :param image: image provided as a NumPy array.
        """
        self.image = image
        self.contour = None   # reset it to none as we have new image

    def initialize_contour(self, method: str = "manual"):
        """
        initialize contour after an image is uploaded.
        
        :param method: "manual" for user-defined points, "auto" for auto-initialization.
        """
        if self.image is None:
            raise ValueError("No image uploaded! Please upload an image first.")

        if method == "auto":
            pass

        elif method == "manual":
            pass

    def evolve_contour(self):
        """
        apply the greedy algorithm to evolve the contour.
        """
        if self.contour is None:
            raise ValueError("Contour not initialized! Call initialize_contour() first.")

        pass
      
    def get_temp_contour(shape: str = "circle") -> np.ndarray: 
    """
    Generate a temporary contour (circle or square) for testing .
    you will need it habiba 
    :param shape: Shape of contour ("circle" or "square").
    :return: NumPy array of contour points.
    """
    if shape == "circle":
        t = np.linspace(0, 2 * np.pi, 100)  
        x = (50 * np.cos(t) + 100).astype(int)  
        y = (50 * np.sin(t) + 100).astype(int)
        return np.column_stack((x, y))

    elif shape == "square":
        return np.array([[50, 50], [150, 50], [150, 150], [50, 150]])  

    def compute_chain_code(self) -> list:
        """
        compute the chain code representation of the contour.
        
        :return: List of chain code directions.
        : habiba you will need an intial contour to work with i will provide you wil generting contours function so you can check on it 
        """
        self.contour = get_temp_contour("circle")
        if self.contour is None:
            raise ValueError("contour not initialized! Call initialize_contour() first.")


        pass

    def compute_perimeter(self) -> float:
        """
        compute the perimeter of the contour.
        
        :return: Perimeter value.
        : habiba you will need an intial contour to work with i will provide you wil generting contours function so you can check on it 
        """
        if self.contour is None:
            raise ValueError("contour not initialized! Call initialize_contour() first.")


        pass

    def compute_area(self) -> float:
        """
        compute the area enclosed by the contour.
        : habiba you will need an intial contour to work with i will provide you wil generting contours function so you can check on it 
        :return: Area value.
        """
        if self.contour is None:
            raise ValueError("vontour not initialized! Call initialize_contour() first.")


        pass

    def visualize_contour(self):
        """
        display the image with the contour overlaid.
        """
        if self.image is None:
            raise ValueError("no image uploaded! please upload an image first.")

        if self.contour is None:
            raise ValueError("contour not initialized! Call initialize_contour() first.")

         # the visualization mechanism to the output viewer 
         pass

