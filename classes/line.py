# from .shape import Shape
import numpy as np
import matplotlib.pyplot as plt
import bisect

class Line():
    def __init__(self, rho, theta, t_range=(-10000, 10000), x_start=0, x_end=500,y_start = 0, y_end = 500, num_of_points=500):
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.rho = rho 
        self.theta = theta
        self.t_range = t_range
        self.num_of_points = num_of_points
        self.__shape_list = []
        self.__fill_shape_list()
    
    def __fill_shape_list(self):
        t = np.linspace(self.t_range[0], self.t_range[1], self.num_of_points)
        x_values = self.rho * np.cos(self.theta) + t * (-np.sin(self.theta))
        y_values = self.rho * np.sin(self.theta) + t * np.cos(self.theta)
        # x_values, y_values = self.trim_list(x_values, y_values, self.x_start)
        self.__shape_list = [x_values, y_values]
        
    def trim_list(self, x_values, y_values, start, end):
        for index, value in enumerate(x_values):
            if value < start or value > end:
                x_values.pop(index)
                y_values.pop(index)
        for index, value in enumerate(y_values):
            if value < start or value > end:
                x_values.pop(index)
                y_values.pop(index)
        return x_values, y_values
        
    @property
    def shape_list(self):
        return self.__shape_list
    
    
if __name__ == "__main__":
    points = [[-253.850622406639, -90.0], [-250.84647302904563, -88.99441340782123], [-249.8450899031812, -88.99441340782123], [-245.8395573997234, -87.98882681564245], [-180.7496542185339, -90.0], [-178.746887966805, -90.0], [-176.74412171507606, -88.99441340782123], [-174.74135546334716, -88.99441340782123], [-172.73858921161826, -87.98882681564245], [-170.73582295988936, -87.98882681564245], [-90.62517289073304, -62.84916201117319], [-88.62240663900417, -61.84357541899441], [-83.61549100968188, -59.832402234636874], [-79.60995850622407, -58.826815642458094], [-0.500691562932218, -90.0], [-0.500691562932218, -88.99441340782123], [-0.500691562932218, -87.98882681564245], [-0.500691562932218, -39.72067039106145], [-0.500691562932218, -38.71508379888268], [-0.500691562932218, -37.70949720670391], [-0.500691562932218, -36.703910614525135], [-0.500691562932218, -35.69832402234637], [-0.500691562932218, -34.6927374301676], [-0.500691562932218, -33.687150837988824], [-0.500691562932218, -32.68156424581005], [-0.500691562932218, -31.67597765363128], [-0.500691562932218, -30.670391061452516], [-0.500691562932218, -29.66480446927374], [-0.500691562932218, -28.659217877094967], [-0.500691562932218, -27.6536312849162], [-0.500691562932218, -25.642458100558656], [-0.500691562932218, -24.636871508379883], [-0.500691562932218, -23.63128491620111], [-0.500691562932218, 90.0], [0.500691562932218, -88.99441340782123], [0.500691562932218, 90.0], [1.5020746887966538, -88.99441340782123], [1.5020746887966538, 88.99441340782124], [2.5034578146611466, 88.99441340782124], [33.54633471645917, 10.558659217877107], [33.54633471645917, 11.564245810055866], [33.54633471645917, 12.569832402234638], [34.547717842323664, 12.569832402234638], [34.547717842323664, 13.575418994413411], [35.5491009681881, 12.569832402234638], [35.5491009681881, 14.581005586592184], [36.550484094052536, 13.575418994413411], [36.550484094052536, 14.581005586592184], [38.553250345781464, 14.581005586592184], [38.553250345781464, 15.586592178770958], [179.74827109266948, 90.0], [183.75380359612723, 88.99441340782124], [245.8395573997234, 13.575418994413411], [246.84094052558783, 15.586592178770958], [247.84232365145226, 13.575418994413411], [247.84232365145226, 14.581005586592184], [248.8437067773167, 12.569832402234638], [248.8437067773167, 13.575418994413411], [248.8437067773167, 14.581005586592184], [248.8437067773167, 16.59217877094973], [249.84508990318113, 14.581005586592184], [250.84647302904568, 14.581005586592184], [253.850622406639, 90.0], [255.85338865836786, 88.99441340782124]]
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x_min, x_max = -1000, 1000
    y_min, y_max = -1000, 1000
    # Load and display the image
    try:
        # Replace 'your_image.jpg' with your image path
        img = plt.imread('data/5.1.11.jpg')
        
        # Get axis limits for proper image placement
        
        # Plot the image as the background
        ax.imshow(img, extent=[x_min, x_max, y_min, y_max], 
                  aspect='auto', alpha=0.5, zorder=0)  # zorder=0 ensures image is behind other elements
    except Exception as e:
        print(f"Error loading image: {e}")
    
    # Plot the lines and points
    for i, point in enumerate(points):
        rho = point[0]
        theta = np.deg2rad(point[1])
        line = Line(rho, theta)
        
        # Plot the line with a label only for the first few to avoid legend clutter
        if i < 5:  # Limit labels for better legend readability
            ax.plot(line.shape_list[0], line.shape_list[1], 
                   label=f"rho={rho}, theta={point[1]}°", zorder=2)
        else:
            ax.plot(line.shape_list[0], line.shape_list[1], zorder=2)
            
        # Plot the point where the normal from origin meets the line
        ax.scatter([rho * np.cos(theta)], [rho * np.sin(theta)], 
                  color='red', s=30, zorder=3)
    
    # Plot a single red point for the legend to avoid duplicates
    ax.scatter([], [], color='red', label="Points (rho, theta)", s=30)
    
    # Add axis lines
    ax.axhline(0, color='black', linewidth=0.5, zorder=1)
    ax.axvline(0, color='black', linewidth=0.5, zorder=1)
    
    # Configure plot
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Line Representation using rho and theta")
    ax.grid(zorder=1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.show()