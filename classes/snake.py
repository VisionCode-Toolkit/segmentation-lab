import numpy as np
import cv2
import matplotlib.pyplot as plt

class ActiveContour:
    def __init__(self, alpha: float = 0.2, beta: float =0.5, gamma: float=1, num_iterations: int=150):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_iterations = num_iterations
        self.image = None  
        self.contour = None 
        self.chain_code = None

    def set_image(self, image: np.ndarray):
        self.image = image

    def set_contour(self, contour: list):
        self.contour = contour

    def evolve_contour(self):
        """
        apply the greedy algorithm to evolve the contour.
        """
        if self.contour is None:
            raise ValueError("Contour not initialized! Call initialize_contour() first.")

        pass

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


