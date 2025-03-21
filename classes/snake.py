import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel


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
    def __init__(self, alpha=2, beta=1, gamma=5, max_iterations=300, window_size=9, sigma=2, mu=0.1, gvf_iterations=80):
        self.alpha = alpha  # elastic
        self.beta = beta   # smooth
        self.gamma = gamma  # external energy
        self.mu = float(mu)  # gvf parameter
        self.gvf_iterations = gvf_iterations  # gvf iterations
        self.max_iterations = max_iterations
        self.window_size = window_size
        self.sigma = sigma  # for blurring
        self.image = None
        self.contour = None
        self.flag_continue = False  # flag to resume
        self.current_iteration = 0

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

    def evolve_step(self):
        # single iteration each 100 ms
        if not self.flag_continue:
            print("Evolution paused.")
            return  # if paused return

        if self.current_iteration >= self.max_iterations:
            print("Contour evolution completed.")
            self.flag_continue = False
            return

        print(f"Iteration {self.current_iteration + 1} started.")

        grad_x, grad_y = self.compute_gradient()
        gvf_x, gvf_y = self.compute_gvf(grad_x, grad_y)
        external_energy = self.compute_external_energy(gvf_x, gvf_y)
        half_win = self.window_size // 2

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
        self.current_iteration += 1

        # print(f"tteration {self.current_iteration} completed.")
        print(f"Window_size {self.window_size}")


        if self.current_iteration >= self.max_iterations:
            print("contour evolution finished.")
            self.flag_continue = False

    def compute_chain_code(self) -> list:
        """
        compute the chain code representation of the contour.
        
        :return: List of chain code directions.
        : habiba you will need an intial contour to work with i will provide you wil generting contours function so you can check on it 
        """
        # self.set_contour()

        if self.contour is None:
            raise ValueError("contour not initialized! Call initialize_contour() first.")
        chain_code = []

        for i in range(1, len(self.contour)):
            current = tuple(self.contour[i])
            prev = tuple(self.contour[i-1])

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
        # self.set_contour()

        if self.contour is None:
            raise ValueError("contour not initialized! Call initialize_contour() first.")
        premeter = 0.0
        for i in range(len(self.contour)):
            current = tuple(self.contour[i])
            next = tuple(self.contour[(i + 1) % len(self.contour)])
            premeter += np.linalg.norm(np.array(current) - np.array(next))

        return premeter


    def compute_area(self) -> float:
        """
        compute the area enclosed by the contour.
        : habiba you will need an intial contour to work with i will provide you wil generting contours function so you can check on it 
        :return: Area value.
        """
        # self.set_contour()
        if self.contour is None:
            raise ValueError("contour not initialized! Call initialize_contour() first.")

        area = 0.0

        for i in range(len(self.contour)):
            x1, y1 = self.contour[i]
            x2, y2 = self.contour[(i + 1) % len(self.contour)]

            area += (x1 * y2) - (x2 * y1)

        return abs(area) / 2.0


