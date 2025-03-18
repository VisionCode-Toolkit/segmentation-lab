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
        self.contour_direction_map = {
            (-1, 0): 0,
            (-1, -1): 1,
            (0, -1): 2,
            (1, -1): 3,
            (1, 0): 4,
            (1, 1): 5,
            (0, 1): 6,
            (-1, 1): 7,
        }

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
        self.set_contour()

        if self.contour is None:
            raise ValueError("contour not initialized! Call initialize_contour() first.")
        chain_code = []
        for i in range(1, len(self.contour)):
            current = tuple(self.contour[i][0])
            prev = tuple(self.contour[i-1][0])

            direction = (current[0] - prev[0], current[1] - prev[1])
            if direction[0] > 0:
                if direction[1] > 0:
                    chain_code.append(5) # Bottom-Left -> prev < current, prev < current
                elif direction[1] < 0:
                    chain_code.append(3) # Top-Left -> prev > current, prev < current
                else:
                    chain_code.append(4) # Left -> prev < current, prev = current
            elif direction[0] < 0:
                if direction[1] > 0:
                    chain_code.append(7) # Bottom-Right -> prev > current, prev < current
                elif direction[1] < 0:
                    chain_code.append(1) # Top-Right -> (prev[0] > current[0], prev[0] > current[0])
                else:
                    chain_code.append(0) # Right -> (prev[0] > current[0], prev[1] = current[1])
            else:
                if direction[1] > 0:
                    chain_code.append(6) # Bottom -> prev = current, prev < current
                elif direction[1] < 0:
                    chain_code.append(2) # Top -> (prev[0] = current[0], prev[0] > current[0])

        return chain_code

    def compute_perimeter(self) -> float:
        """
        compute the perimeter of the contour.
        
        :return: Perimeter value.
        : habiba you will need an intial contour to work with i will provide you wil generting contours function so you can check on it 
        """
        self.set_contour()

        if self.contour is None:
            raise ValueError("contour not initialized! Call initialize_contour() first.")
        premeter = 0.0
        for i in range(len(self.contour)):
            current = tuple(self.contour[i][0])
            next = tuple(self.contour[(i + 1) % len(self.contour)][0])
            premeter += np.linalg.norm(np.array(current) - np.array(next))

        return premeter


    def compute_area(self) -> float:
        """
        compute the area enclosed by the contour.
        : habiba you will need an intial contour to work with i will provide you wil generting contours function so you can check on it 
        :return: Area value.
        """
        self.set_contour()
        if self.contour is None:
            raise ValueError("contour not initialized! Call initialize_contour() first.")

        area = 0.0

        for i in range(len(self.contour)):
            x1, y1 = self.contour[i][0]
            x2, y2 = self.contour[(i + 1) % len(self.contour)][0]

            area += (x1 * y2) - (x2 * y1)

        return abs(area) / 2.0


