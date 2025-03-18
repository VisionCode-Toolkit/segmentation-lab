import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel


class ActiveContour:
    def __init__(self, alpha=2, beta=1, gamma=5, max_iterations=300, window_size=9, sigma=2, mu=0.1, gvf_iterations=80):
        self.alpha = alpha  # Elasticity term
        self.beta = beta  # Smoothness term
        self.gamma = gamma  # Weight of external energy
        self.mu = float(mu)  # GVF regularization parameter
        self.gvf_iterations = gvf_iterations  # Number of GVF iterations
        self.max_iterations = max_iterations
        self.window_size = window_size  # Size of search window
        self.sigma = sigma  # Gaussian smoothing parameter
        self.image = None
        self.contour = None
        self.flag_continue = False  # Flag to pause/resume evolution

    def set_image(self, image):

        if image.shape[-1] == 3:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.image = image
        self.image = self.image.astype(np.float32)

    def set_contour(self, contour):
        self.contour = np.array(contour, dtype=np.float32)

    def compute_gradient(self):
        blurred = gaussian_filter(self.image, self.sigma)
        edges = cv2.Canny(self.image.astype(np.uint8), 100, 200)
        grad_x = cv2.Sobel(edges, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(edges, cv2.CV_32F, 0, 1, ksize=3)
        return grad_x, grad_y

    def compute_gvf(self, grad_x, grad_y):
        u, v = grad_x.copy(), grad_y.copy()
        for _ in range(self.gvf_iterations):
            u += self.mu * (cv2.Laplacian(u, cv2.CV_32F)) - (u - grad_x)
            v += self.mu * (cv2.Laplacian(v, cv2.CV_32F)) - (v - grad_y)
        return u, v

    def compute_external_energy(self, gvf_x, gvf_y):
        magnitude = np.sqrt(gvf_x**2 + gvf_y**2) + 1e-6
        return -magnitude

    def compute_internal_energy(self, contour):
        diff1 = np.diff(contour, axis=0, append=contour[:1])  # First derivative
        diff2 = np.diff(diff1, axis=0, append=diff1[:1])  # Second derivative
        elastic_energy = np.linalg.norm(diff1, axis=1) ** 2
        smoothness_energy = np.linalg.norm(diff2, axis=1) ** 2
        return self.alpha * elastic_energy + self.beta * smoothness_energy

    def evolve_contour(self):
        print("Starting contour evolution...")
        if self.contour is None:
            raise ValueError("Contour not initialized! Call set_contour() first.")


        grad_x, grad_y = self.compute_gradient()
        gvf_x, gvf_y = self.compute_gvf(grad_x, grad_y)
        external_energy = self.compute_external_energy(gvf_x, gvf_y)
        half_win = self.window_size // 2

        for iteration in range(self.max_iterations):
            if not self.flag_continue:
                print(f"Evolution paused at iteration {iteration}")
                break

            new_contour = np.copy(self.contour)
            internal_energy = self.compute_internal_energy(self.contour)

            for i, (x, y) in enumerate(self.contour):
                x, y = int(x), int(y)
                search_window = [(x + dx, y + dy) for dx in range(-half_win, half_win + 1)
                                 for dy in range(-half_win, half_win + 1)
                                 if 0 <= x + dx < self.image.shape[1] and 0 <= y + dy < self.image.shape[0]]

                energies = []
                for nx, ny in search_window:
                    candidate_contour = np.copy(new_contour)
                    candidate_contour[i] = [nx, ny]
                    total_energy = (np.sum(self.compute_internal_energy(candidate_contour)) +
                                    self.gamma * external_energy[int(ny), int(nx)])
                    energies.append(total_energy)

                min_idx = np.argmin(energies)
                new_contour[i] = search_window[min_idx]

            self.contour = new_contour
            print(f"Iteration {iteration + 1} completed.")

        print("Contour evolution finished.")
        print(f"n of iterations {self.max_iterations}, alpha is {self.alpha}, beta is {self.beta}, gamma is {self.gamma}")

    def compute_chain_code(self) -> list:
        """
        compute the chain code representation of the contour.
        
        :return: List of chain code directions.
        : habiba you will need an intial contour to work with i will provide you wil generting contours function so you can check on it 
        """

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


