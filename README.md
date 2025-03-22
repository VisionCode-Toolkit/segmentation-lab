# segmentation-lab



 ## Description
This repository contains an implementation of edge and boundary detection techniques **from scratch**, including:
- **Canny edge detection**
- **Hough Transform** for detecting lines, circles, and ellipses
- **Active Contour Model (Snake) using the greedy algorithm**
- **Chain code representation & computation of perimeter and area**

##  Features
 **Edge detection using Canny (Implemented from scratch)** 
 **Hough Transform for detecting geometric shapes**  
   - **Lines** 
   - **Circles** 
   - **Ellipses** 
 **Active Contour Model (Snake) for object segmentation** 
 **Contour representation using chain code**  
 **Computing shape properties (perimeter & area)**  


---

##  Implementation Details
###  Edge Detection & Shape Extraction
#### Canny Edge Detector (Implemented from scratch)

The Canny edge detector is a multi-stage algorithm that involves:
1. **Noise Reduction**: Gaussian filtering to smooth the image and reduce noise.
2. **Gradient Calculation**: Compute intensity gradients using Sobel filters.
3. **Non-maximum Suppression**: Thin out edges to retain only the most significant ones.
4. **Double Thresholding & Edge Tracking**: Classify edges based on intensity and track strong edges.

#### Hough Transform for Shape Detection
The Hough Transform is a feature extraction technique used to detect simple shapes:
- **Lines**: Convert points in Cartesian space to curves in Hough space and find intersections.
- **Circles**: Accumulate votes in a parameter space to identify circle centers and radii.
- **Ellipses**: Extend the concept of circle detection by adding extra parameters for eccentricity.


- **Lines**: [Video](https://your-video-link.com)
- **Circles**: [Video](https://your-video-link.com)
- **Ellipses**: [Video](https://your-video-link.com)

###  Active Contour Model (Snake)

The **Active Contour Model (Snake)** is an energy-minimizing spline that evolves toward object boundaries:
- **Initialization**: Define an initial contour close to the target object.
- **Energy Minimization**:
  - **Internal Energy**: Maintains smoothness by penalizing sharp bends.
  - **External Energy**: Attracts the contour toward object edges using image gradients.
  - **Greedy Algorithm**: Iteratively adjusts contour points to minimize total energy.
- **Chain Code Representation**: Represent contours as directional codes for efficient shape analysis.
- **Perimeter & Area Calculation**: Extract geometric properties from the final contour.

---

## Dependencies

The segmentation  lab relies on the following technologies and libraries:

| **Dependency**       | **Description**                                       |
|-----------------------|-------------------------------------------------------|
| Python 3.x           | Core programming language.                            |
| NumPy                | Numerical computations for signal processing.         |
| PyQt5                | GUI framework for building desktop applications.      |
| pyqtgraph            | Fast plotting and 2D visualization in PyQt.           |
| matplotlib           | Visualization library for plotting and analysis.      |
| OpenCV (cv2)         | Computer vision library for image manipulation.       |        |


## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/Mostafaali3/segmentation-lab
   cd segmentation-lab
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python main.py
   ```
   ## Contributors
<div align="center">
  <table style="border-collapse: collapse; border: none;">
    <tr>
      <td align="center" style="border: none;">
        <img src="https://avatars.githubusercontent.com/Mostafaali3" alt="Mostafa Ali" width="150" height="150"><br>
        <a href="https://github.com/Mostafaali3"><b>Mostafa Ali</b></a>
      </td>
      <td align="center" style="border: none;">
        <img src="https://avatars.githubusercontent.com/habibaalaa123" alt="Habiba Alaa" width="150" height="150"><br>
        <a href="https://github.com/habibaalaa123"><b>Habiba Alaa</b></a>
      </td>
      <td align="center" style="border: none;">
        <img src="https://avatars.githubusercontent.com/enjyashraf18" alt="Anjy Ashraf" width="150" height="150"><br>
        <a href="https://github.com/enjyashraf18"><b>Enjy Ashraf</b></a>
      </td>
      </td>
      <td align="center" style="border: none;">
        <img src="https://avatars.githubusercontent.com/Shahd-A-Mahmoud" alt="Shahd Ahmed" width="150" height="150"><br>
        <a href="https://github.com/Shahd-A-Mahmoud"><b>Shahd Ahmed</b></a>
      </td>
  </table>
</div>

## License
This project is open-source and available under the [MIT License](LICENSE).


