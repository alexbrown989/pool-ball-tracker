import cv2
import numpy as np
from typing import List, Tuple

class BallDetector:

    """
    Detects pool balls using computer vision techniques.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Load settings from config.yaml.
        # We need to know how big a ball is (min_radius, max_radius)
        # to distinguish it from noise.
        self.min_radius = config['ball_detection']['min_radius']
        self.max_radius = config['ball_detection']['max_radius']
        
        # Calculate Area (Circle Math: pi * r^2).
        self.min_area = (np.pi * self.min_radius**2) * 0.5
        self.max_area = (np.pi * self.max_radius**2) * 1.5
        
        # 0.3 allows for ovals, 1.4 for glare.
        self.min_circ = 0.3
        self.max_circ = 1.4
        
        # Crop settings: We don't want to look at the floor or the crowd,
        # only the pool table.
        self.crop_top = config['video'].get('crop', {}).get('top', 0)
        self.crop_bottom = config['video'].get('crop', {}).get('bottom', 0)
        self.crop_left = config['video'].get('crop', {}).get('left', 0)
        self.crop_right = config['video'].get('crop', {}).get('right', 0)
        
        self.debug = False # Turn this on to see print statements describing what's happening.

    def set_debug(self, debug_on: bool):
        """Externally set the debug mode."""
        self.debug = debug_on
        
    def detect_balls(self, frame) -> List[Tuple[int, int, int]]:
        """
        The main vision recipe.
        Returns a list of balls: [(x, y, radius), (x, y, radius), ...]
        """
        h, w = frame.shape[:2]
        
        # 1. CROP: Cut out the part of the image we don't care about.
        cropped = frame[self.crop_top:h-self.crop_bottom, 
                          self.crop_left:w-self.crop_right]
        # If crop failed (image is empty), give up.
        if cropped.size == 0:
            if self.debug: print("ERROR: Frame was cropped to nothing. Check crop settings.")
            return []
        
        # 2. COLOR CONVERSION: Change from Blue-Green-Red (standard) to HSV.
        # HSV (Hue, Saturation, Value) is much better for detecting color.
        try:
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        except cv2.error as e:
            print(f"ERROR: cvtColor failed! {e}")
            return []

        # 3. MASKING: Find the Felt.
        # We look for the blue/green color of the table felt.
        felt_lower = np.array(self.config['ball_detection']['felt_color']['lower'])
        felt_upper = np.array(self.config['ball_detection']['felt_color']['upper'])

        # 'inRange' creates a black-and-white image.
        # White = This pixel matches the felt color.
        # Black = This pixel does NOT match.
        mask_felt = cv2.inRange(hsv, felt_lower, felt_upper)
        # INVERT: We want the BALLS (not felt) to be White.
        mask_objects = cv2.bitwise_not(mask_felt)
        
        # 4. CLEAN UP: "Morphology".
        # This removes tiny specks of white noise (salt) and fills in small holes.
        kernel = np.ones((3, 3), np.uint8)
        mask_opened = cv2.morphologyEx(mask_objects, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 5. CONTOURS: Find the outlines of the white shapes.
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_balls = []

        if self.debug:
            print("\n--- Filtering Contours (Debug Mode) ---")
            print(f"Found {len(contours)} total contours after cleaning.")
            print(f"Filters (DYNAMIC): Area ({self.min_area:.1f}-{self.max_area:.1f}), Radius ({self.min_radius}-{self.max_radius}), Circ ({self.min_circ}-{self.max_circ})")
        
        # 6. FILTER: Loop through every shape we found and check if it's a ball.
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Check 1: Is the size correct?
            if not (self.min_area < area < self.max_area):
                if self.debug: print(f"Contour {i}: REJECTED (Area = {area:.2f})")
                continue # Too big or too small. Skip it.
            # Get the smallest circle that fits around this shape.
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # Check 2: Is the radius correct?
            if not (self.min_radius < radius < self.max_radius):
                if self.debug: print(f"Contour {i}: Area OK. REJECTED (Radius = {radius:.2f})")
                continue # Radius doesn't match a pool ball.
            
            # Check 3: Is it actually a circle?
            # We compare the Area to a perfect circle's area.
            circularity = area / (np.pi * (radius ** 2)) if radius > 0 else 0
            if not (self.min_circ < circularity < self.max_circ):
                if self.debug: print(f"Contour {i}: Area/Rad OK. REJECTED (Circularity = {circularity:.2f})")
                continue # It's a weird shape (like a chalk cube).

            if self.debug: print(f"Contour {i}: PASSED basic filters. Area={area:.2f}, Radius={radius:.2f}, Circ={circularity:.2f}. Verifying...")

            # If we passed all checks, calculate the REAL coordinates.
            # (We have to add the crop values back to get the position on the full screen).
            center_x = int(x) + self.crop_left
            center_y = int(y) + self.crop_top
            radius_int = int(radius)
            
            # Check 4: Final verification (check brightness).
            if self._verify_ball(frame, center_x, center_y, radius_int):
                if self.debug: print(f"Contour {i}: -> ✅ ACCEPTED")
                detected_balls.append((center_x, center_y, radius_int))
            else:
                if self.debug: print(f"Contour {i}: -> REJECTED (Failed _verify_ball check)")
        
        if self.debug: print("--- Filtering Complete ---")
        
        return detected_balls
    
    def _verify_ball(self, frame, x, y, r) -> bool:
        """
        Double check: Is the spot bright enough?
        This prevents detecting shadows or pockets as balls.
        """
        h, w = frame.shape[:2]
        
        # Don't check if it's on the very edge of the screen.
        if x - r < 5 or x + r >= w - 5 or y - r < 5 or y + r >= h - 5:
            return False 

        # Create a small mask for just this one ball.
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (x, y), max(1, r - 1), 255, -1)

        # Get the average brightness (Value in HSV) of this circle.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.mean(hsv, mask=mask)[:3]
        brightness = mean_hsv[2]

        # If it's too dark (less than 40 brightness), it's probably a shadow.
        if brightness < 40: 
            return False
        
        return True
    
    def is_white_ball(self, frame, x, y, r) -> bool:
        """Check if ball is white cue ball."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), max(1, r - 1), 255, -1)
        
        mean_hsv = cv2.mean(hsv, mask=mask)[:3]
        
        lower = np.array(self.config['ball_detection']['white_ball']['lower'])
        upper = np.array(self.config['ball_detection']['white_ball']['upper'])

        if (lower[1] <= mean_hsv[1] <= upper[1] and 
            lower[2] <= mean_hsv[2] <= upper[2]):
            return True
        
        return False
