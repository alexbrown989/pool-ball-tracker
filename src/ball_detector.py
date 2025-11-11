import cv2
import numpy as np
from typing import List, Tuple


class BallDetector:
    """Detects pool balls in video frames using color segmentation and Hough circles."""
    
    def __init__(self, config):
        self.config = config
        self.min_radius = config['ball_detection']['min_radius']
        self.max_radius = config['ball_detection']['max_radius']
        
    def preprocess_frame(self, frame):
        """Apply preprocessing to improve ball detection."""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(hsv, (9, 9), 2)
        
        return blurred
    
    def detect_balls(self, frame) -> List[Tuple[int, int, int]]:
        """
        Detect all balls in the frame.
        
        Returns:
            List of (x, y, radius) tuples for detected balls
        """
        preprocessed = self.preprocess_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred_gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        detected_balls = []
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            for circle in circles[0, :]:
                x, y, r = circle
                
                # Verify it's actually a ball by checking if it's roughly circular
                # and has reasonable color properties
                if self._verify_ball(frame, x, y, r):
                    detected_balls.append((int(x), int(y), int(r)))
        
        return detected_balls
    
    def _verify_ball(self, frame, x, y, r) -> bool:
        """Verify that a detected circle is actually a ball."""
        # Make sure coordinates are within frame bounds
        h, w = frame.shape[:2]
        if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
            return False
        
        # Extract the circular region
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Check if the region has ball-like properties
        # (relatively uniform color, not too dark, not the table felt)
        roi = frame[max(0, y-r):min(h, y+r), max(0, x-r):min(w, x+r)]
        
        if roi.size == 0:
            return False
        
        # Calculate mean color
        mean_color = cv2.mean(roi)[:3]
        
        # Balls should have some brightness (not pure shadows)
        if sum(mean_color) / 3 < 30:
            return False
        
        return True
    
    def is_white_ball(self, frame, x, y, r) -> bool:
        """Check if a detected ball is the white cue ball."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Extract the ball region
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Get the mean HSV values
        mean_hsv = cv2.mean(hsv, mask=mask)[:3]
        
        # Check if it matches white ball criteria
        lower = np.array(self.config['ball_detection']['white_ball']['lower'])
        upper = np.array(self.config['ball_detection']['white_ball']['upper'])
        
        if (lower[0] <= mean_hsv[0] <= upper[0] and 
            lower[1] <= mean_hsv[1] <= upper[1] and 
            lower[2] <= mean_hsv[2] <= upper[2]):
            return True
        
        return False
