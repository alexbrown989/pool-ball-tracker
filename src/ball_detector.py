import cv2
import numpy as np
from typing import List, Tuple


class BallDetector:
    """Detects pool balls using color segmentation - more robust than HoughCircles."""
    
    def __init__(self, config):
        self.config = config
        self.min_radius = config['ball_detection']['min_radius']
        self.max_radius = config['ball_detection']['max_radius']
        
    def detect_balls(self, frame) -> List[Tuple[int, int, int]]:
        """
        Detect balls using color segmentation to isolate them from table felt.
        Much more robust than HoughCircles for varied lighting and motion.
        
        Returns:
            List of (x, y, radius) tuples for detected balls
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Mask for table felt (blue in your case)
        # Adjust these values if needed for your specific table
        lower_felt = np.array([90, 50, 50])   # Blue felt
        upper_felt = np.array([130, 255, 255])
        
        # Create mask of the felt
        mask_felt = cv2.inRange(hsv, lower_felt, upper_felt)
        
        # Invert to get everything BUT the felt (balls, pockets, rails)
        mask_objects = cv2.bitwise_not(mask_felt)
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask_objects = cv2.morphologyEx(mask_objects, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_objects = cv2.morphologyEx(mask_objects, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask_objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_balls = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (balls should be within this range)
            if not (200 < area < 3000):
                continue
            
            # Get minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # Check radius is reasonable
            if not (self.min_radius < radius < self.max_radius):
                continue
            
            # Check circularity (balls are round)
            circularity = area / (np.pi * (radius ** 2))
            
            if not (0.65 < circularity < 1.3):  # Allow some tolerance
                continue
            
            center_x, center_y = int(x), int(y)
            radius = int(radius)
            
            # Final verification - reject pockets and false positives
            if self._verify_ball(frame, center_x, center_y, radius):
                detected_balls.append((center_x, center_y, radius))
        
        return detected_balls
    
    def _verify_ball(self, frame, x, y, r) -> bool:
        """Verify detected circle is a ball, not a pocket or false positive."""
        h, w = frame.shape[:2]
        
        # Check bounds
        if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
            return False
        
        # Create circular mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Get mean color in HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.mean(hsv, mask=mask)[:3]
        
        brightness = mean_hsv[2]  # V channel
        saturation = mean_hsv[1]  # S channel
        
        # CRITICAL: Reject dark pockets
        if brightness < 70:
            return False
        
        # Reject very dark corners (pockets)
        margin = 100
        is_corner = ((x < margin and y < margin) or 
                     (x > w - margin and y < margin) or
                     (x < margin and y > h - margin) or
                     (x > w - margin and y > h - margin))
        
        if is_corner and brightness < 90:
            return False
        
        # Check uniformity - balls are solid colors
        roi = frame[max(0, y-r):min(h, y+r), max(0, x-r):min(w, x+r)]
        if roi.size == 0:
            return False
        
        std_dev = np.std(roi)
        if std_dev > 55:  # Too much variation
            return False
        
        return True
    
    def is_white_ball(self, frame, x, y, r) -> bool:
        """Check if detected ball is the white cue ball."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        mean_hsv = cv2.mean(hsv, mask=mask)[:3]
        
        # White ball has low saturation and high brightness
        if mean_hsv[1] < 40 and mean_hsv[2] > 180:
            return True
        
        return False