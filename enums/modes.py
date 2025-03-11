from enum import Enum 

class Modes(Enum):
    CANNY = 'Canny Detector'
    LINE = 'Hough - Line Detection'
    CIRCLE = 'Hough - Circle Detection'
    ELLIPSE = 'Hough - Ellipse Detection'
    SNAKE = 'Active Contour (Snake)'