import cv2
import numpy as np
import sys
import time
import threading
from collections import deque

class ArUcoDetector:
    def __init__(self):
        """Initialize ArUco detector with performance optimizations"""
        # Use fewer, most common dictionaries for better performance
        self.aruco_dicts = {
            '4x4_50': cv2.aruco.DICT_4X4_50,
            '5x5_50': cv2.aruco.DICT_5X5_50,
            '6x6_50': cv2.aruco.DICT_6X6_50,
        }
        
        # Optimized detector parameters for speed
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 15
        self.parameters.adaptiveThreshWinSizeStep = 4
        self.parameters.minMarkerPerimeterRate = 0.05
        self.parameters.maxMarkerPerimeterRate = 2.0
        self.parameters.polygonalApproxAccuracyRate = 0.05
        
        self.cap = None
        self.frame_buffer = deque(maxlen=2)  # Small buffer for smoothing
        self.fps_counter = deque(maxlen=30)  # For FPS calculation
        
        # Camera calibration parameters for depth estimation (approximate values)
        # These are rough estimates - for precise measurements, camera calibration is needed
        self.camera_matrix = None
        self.dist_coeffs = None
        self.marker_size_cm = 5.0  # Assumed marker size in cm (adjust as needed)
        
    def initialize_camera(self):
        """Initialize camera with high performance settings"""
        print("🚀 Initializing high-performance camera...")
        
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        
        for backend in backends:
            for camera_idx in range(3):
                try:
                    cap = cv2.VideoCapture(camera_idx, backend)
                    
                    if cap.isOpened():
                        time.sleep(0.5)  # Shorter wait time
                        
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            print(f"✅ Camera {camera_idx} opened!")
                            
                            # High performance settings
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                            cap.set(cv2.CAP_PROP_FPS, 60)  # Request higher FPS
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
                            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                            
                            # Set approximate camera matrix for depth estimation
                            # These are estimates for a typical laptop camera
                            fx = fy = 800  # Focal length approximation
                            cx, cy = 640, 360  # Principal point (image center)
                            self.camera_matrix = np.array([[fx, 0, cx],
                                                         [0, fy, cy],
                                                         [0, 0, 1]], dtype=np.float32)
                            self.dist_coeffs = np.zeros((4, 1))  # Assume no distortion
                            
                            self.cap = cap
                            return True
                        
                    cap.release()
                except Exception as e:
                    continue
        
        print("❌ Camera initialization failed")
        return False
    
    def detect_markers_fast(self, frame):
        """Optimized marker detection"""
        if frame is None:
            return []
            
        # Convert to grayscale once
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply slight gaussian blur to reduce noise (improves detection)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        detected_markers = []
        
        # Try only the most common dictionaries first
        priority_dicts = ['4x4_50', '5x5_50']
        
        for dict_name in priority_dicts:
            try:
                dict_type = self.aruco_dicts[dict_name]
                aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
                detector = cv2.aruco.ArucoDetector(aruco_dict, self.parameters)
                corners, ids, _ = detector.detectMarkers(gray)
                
                if ids is not None:
                    for i in range(len(ids)):
                        detected_markers.append({
                            'corners': corners[i],
                            'id': int(ids[i][0]),
                            'dict': dict_name,
                            'center': corners[i][0].mean(axis=0)
                        })
                        
            except Exception:
                continue
        
        # If no markers found, try remaining dictionaries
        if not detected_markers:
            for dict_name in ['6x6_50']:
                try:
                    dict_type = self.aruco_dicts[dict_name]
                    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
                    detector = cv2.aruco.ArucoDetector(aruco_dict, self.parameters)
                    corners, ids, _ = detector.detectMarkers(gray)
                    
                    if ids is not None:
                        for i in range(len(ids)):
                            detected_markers.append({
                                'corners': corners[i],
                                'id': int(ids[i][0]),
                                'dict': dict_name,
                                'center': corners[i][0].mean(axis=0)
                            })
                            
                except Exception:
                    continue
        
        return detected_markers
    
    def estimate_depth_and_distance(self, marker):
        """Estimate marker depth from camera and distance from center in cm"""
        corners = marker['corners'][0]
        
        # Calculate marker size in pixels (diagonal of the marker)
        marker_diagonal_pixels = np.linalg.norm(corners[0] - corners[2])
        
        # Estimate depth using marker size (rough approximation)
        # This assumes marker diagonal is ~7cm (5cm marker with some margin)
        marker_diagonal_cm = self.marker_size_cm * 1.414  # diagonal = side * sqrt(2)
        
        # Simple depth estimation: depth = (focal_length * real_size) / pixel_size
        focal_length = self.camera_matrix[0, 0] if self.camera_matrix is not None else 800
        depth_cm = (focal_length * marker_diagonal_cm) / marker_diagonal_pixels
        
        # Convert distance from center from pixels to approximate cm
        # This uses the depth to scale the pixel distance
        center = marker['center']
        screen_center = np.array([640, 360])  # Approximate screen center
        pixel_distance = np.linalg.norm(center - screen_center)
        
        # Convert pixel distance to cm using depth scaling
        pixels_per_cm_at_depth = focal_length / depth_cm
        distance_from_center_cm = pixel_distance / pixels_per_cm_at_depth
        
        return depth_cm, distance_from_center_cm
    
    def draw_centering_guide(self, frame):
        """Draw centering guide overlay"""
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Draw center crosshair
        cross_size = 30
        thickness = 2
        
        # Main crosshair
        cv2.line(frame, (center_x - cross_size, center_y), 
                (center_x + cross_size, center_y), (0, 255, 255), thickness)
        cv2.line(frame, (center_x, center_y - cross_size), 
                (center_x, center_y + cross_size), (0, 255, 255), thickness)
        
        # Center point
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
        
        # Corner guides
        corner_size = 20
        corner_offset = 100
        
        corners = [
            (center_x - corner_offset, center_y - corner_offset),  # Top-left
            (center_x + corner_offset, center_y - corner_offset),  # Top-right
            (center_x - corner_offset, center_y + corner_offset),  # Bottom-left
            (center_x + corner_offset, center_y + corner_offset),  # Bottom-right
        ]
        
        for corner_x, corner_y in corners:
            cv2.line(frame, (corner_x - corner_size//2, corner_y), 
                    (corner_x + corner_size//2, corner_y), (100, 100, 255), 1)
            cv2.line(frame, (corner_x, corner_y - corner_size//2), 
                    (corner_x, corner_y + corner_size//2), (100, 100, 255), 1)
        
        return frame
    
    def draw_tracking_arrows(self, frame, markers):
        """Draw directional arrows pointing to screen center"""
        if not markers:
            return frame
            
        height, width = frame.shape[:2]
        screen_center = np.array([width // 2, height // 2])
        
        for marker in markers:
            marker_center = marker['center'].astype(int)
            
            # Calculate direction vector from marker to screen center
            direction = screen_center - marker_center
            distance_pixels = np.linalg.norm(direction)
            
            # Get depth and distance in cm
            depth_cm, distance_cm = self.estimate_depth_and_distance(marker)
            
            # Only show arrows if marker is not centered
            if distance_pixels > 15:  # Threshold in pixels
                # Normalize direction and create arrow
                direction_normalized = direction / distance_pixels
                
                # Arrow properties
                arrow_length = min(100, distance_pixels * 0.4)
                arrow_start = marker_center
                arrow_end = (arrow_start + direction_normalized * arrow_length).astype(int)
                
                # Color based on distance (green when close, red when far)
                color_intensity = min(255, int(distance_cm * 30))
                arrow_color = (0, 255 - color_intensity // 3, color_intensity)
                
                # Draw arrow pointing directly to center
                cv2.arrowedLine(frame, tuple(arrow_start), tuple(arrow_end), 
                              arrow_color, 4, tipLength=0.4)
                
                # Distance from center in cm
                cv2.putText(frame, f"{distance_cm:.1f}cm", 
                           (marker_center[0] + 15, marker_center[1] - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)
                
                # Depth from camera
                cv2.putText(frame, f"Depth: {depth_cm:.0f}cm", 
                           (marker_center[0] + 15, marker_center[1] + 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # Marker is centered
                depth_cm, _ = self.estimate_depth_and_distance(marker)
                cv2.putText(frame, "CENTERED!", 
                           (marker_center[0] - 50, marker_center[1] - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Depth: {depth_cm:.0f}cm", 
                           (marker_center[0] - 50, marker_center[1] + 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_markers_optimized(self, frame, markers):
        """Draw detected markers with optimized rendering"""
        for marker in markers:
            corners = marker['corners'][0].astype(int)
            marker_id = marker['id']
            dict_name = marker['dict']
            center = marker['center'].astype(int)
            
            # Draw marker outline (thicker for better visibility)
            cv2.polylines(frame, [corners], True, (0, 255, 0), 3)
            
            # Draw marker info with better positioning
            cv2.putText(frame, f"ID: {marker_id}", 
                       (center[0] - 35, center[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, dict_name, 
                       (center[0] - 35, center[1] + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Draw corners with different colors
            corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            for i, corner in enumerate(corners):
                cv2.circle(frame, tuple(corner), 4, corner_colors[i % 4], -1)
            
            # Draw center point
            cv2.circle(frame, tuple(center), 5, (255, 0, 255), -1)
        
        return frame
    
    def draw_performance_overlay(self, frame, fps, markers):
        """Draw performance and status information"""
        # Performance info
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Markers: {len(markers)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        instructions = [
            "Center marker in yellow crosshair",
            "Arrows point to center | Distance in cm", 
            "Depth shows distance from camera",
            "'q': Quit | 's': Save | 'g': Toggle guide"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = frame.shape[0] - (len(instructions) - i) * 25
            cv2.putText(frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main high-performance detection loop"""
        print("\n🚀 High-Performance ArUco Detector with Centering Guide")
        print("=" * 55)
        print(f"OpenCV Version: {cv2.__version__}")
        
        if not self.initialize_camera():
            return
        
        print("\n📹 High-FPS camera ready!")
        print("🎯 Use the centering guide to align markers")
        
        show_guide = True
        frame_times = deque(maxlen=10)
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    continue
                
                # Fast marker detection
                markers = self.detect_markers_fast(frame)
                
                # Draw centering guide
                if show_guide:
                    frame = self.draw_centering_guide(frame)
                
                # Draw tracking arrows
                frame = self.draw_tracking_arrows(frame, markers)
                
                # Draw markers
                frame = self.draw_markers_optimized(frame, markers)
                
                # Calculate FPS
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                avg_frame_time = sum(frame_times) / len(frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                
                # Draw overlay
                frame = self.draw_performance_overlay(frame, fps, markers)
                
                # Show frame
                cv2.imshow('ArUco Detector - High FPS with Centering Guide', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f'aruco_centered_{int(time.time())}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"📸 Saved: {filename}")
                elif key == ord('g'):
                    show_guide = not show_guide
                    print(f"🎯 Guide {'ON' if show_guide else 'OFF'}")
                
        except KeyboardInterrupt:
            print("\n⚡ Stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🔒 Cleanup complete")

def main():
    try:
        if not hasattr(cv2, 'aruco'):
            raise ImportError("ArUco module not found")
        print("✅ OpenCV with ArUco support ready")
    except ImportError:
        print("❌ Install: pip install opencv-contrib-python")
        return
    
    detector = ArUcoDetector()
    detector.run()

if __name__ == "__main__":
    main()