# High-Performance ArUco Marker Detector for Autonomous Precision Landing

**Organization:** Skye Air Mobility Pvt Ltd.  
**Project:** Research and Development for Autonomous Precision Landing Systems  
**Version:** 1.0.0

---

## 1. Overview

This project is a high-performance ArUco marker detection and tracking system developed as part of a research and development internship at Skye Air Mobility. The primary goal of this tool is to provide robust, real-time positional data of a target landing zone, identified by an ArUco marker.

The system uses OpenCV to capture high-framerate video, detects various types of ArUco markers, and provides crucial metadata such as the marker's ID, its depth from the camera, and its distance from the camera's center point. This information serves as the foundational perception input for an autonomous drone precision landing algorithm.



The script includes an interactive visual guide to help in aligning the marker, making it a valuable tool for testing, calibration, and development of landing logic.

## 2. Key Features

* **High-Performance Camera Handling:** Initializes the camera for high resolution (1280x720) and high FPS (60 FPS target) for minimal latency.
* **Optimized Multi-Dictionary Detection:** Efficiently scans for markers from the most common ArUco dictionaries (`4x4`, `5x5`, `6x6`).
* **Real-time Depth and Distance Estimation:** Provides approximate measurements in centimeters of the marker's depth (distance from camera) and its offset from the screen center.
* **Interactive Centering Guide:** An on-screen overlay with a crosshair and guides to help manually or automatically center the marker.
* **Visual Tracking Arrows:** Directional arrows are drawn on-screen, pointing from the marker towards the center, with color-coding to indicate distance.
* **Comprehensive Data Overlay:** Displays real-time FPS, number of detected markers, and clear user instructions on the video feed.

## 3. Setup and Installation

Follow these steps to set up and run the ArUco detector on your system.

### Prerequisites

* Python 3.7+
* A webcam connected to your system.

### Installation Steps

1.  **Clone the Repository (or download the script):** If this project is part of a Git repository, clone it. Otherwise, save the `aruco_detector.py` file to a local directory.

2.  **Install Required Libraries:** This project requires `OpenCV` and `NumPy`. Open your terminal or command prompt and run the following command to install the necessary packages:

    ```bash
    pip install opencv-contrib-python numpy
    ```
    *Note: `opencv-contrib-python` is required as the core ArUco functionalities are included in the contrib modules.*

## 4. How to Run the Detector

Once the setup is complete, you can run the script directly from your terminal.

1.  Navigate to the directory where you saved `aruco_detector.py`.
2.  Run the following command:
    ```bash
    python aruco_detector.py
    ```
3.  A window titled "ArUco Detector - High FPS with Centering Guide" should appear, showing your webcam feed. If you have multiple cameras, the script will attempt to find the first available one.
4.  Point your camera at an ArUco marker to see the detection in action.

## 5. Usage and Controls

The application provides real-time visual feedback and can be controlled with simple keyboard commands.

### On-Screen Information

* **Marker Outline:** A green polygon is drawn around any detected marker.
* **Marker ID & Dictionary:** The ID and dictionary type (e.g., `4x4_50`) are displayed near the marker.
* **Tracking Arrow:** An arrow points from the marker's center to the frame's center. Its color and length indicate how far off-center the marker is.
* **Distance from Center:** The text next to the arrow shows the estimated distance (in cm) from the center.
* **Depth:** The estimated distance (in cm) from the camera to the marker.
* **Performance Stats:** The top-left corner displays the current Frames Per Second (FPS) and the total number of markers detected in the frame.

### Keyboard Controls

* **`q`**: Press 'q' to quit the application and close the window.
* **`s`**: Press 's' to save the current frame as a `.jpg` file in the script's directory. This is useful for documentation and analysis.
* **`g`**: Press 'g' to toggle the yellow centering guide overlay on and off.

---
*Developed for Skye Air Mobility Pvt Ltd. (R&D)*
