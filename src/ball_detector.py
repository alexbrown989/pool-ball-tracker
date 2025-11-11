import cv2
import numpy as np
from typing import List, Tuple

class BallDetector:
    """
    Detects pool balls using a robust, config-driven,
    and commented pipeline.
    """
    
    def __init__(self, config):
        self.config = config
        
        # ---
        # DYNAMIC FILTERS: Driven by config.yaml
        # This is your *most important* setting.
        # ---
        self.min_radius = config['ball_detection']['min_radius']
        self.max_radius = config['ball_detection']['max_radius']
        
        # We derive Area from Radius, giving 50% tolerance
        self.min_area = (np.pi * self.min_radius**2) * 0.5
        self.max_area = (np.pi * self.max_radius**2) * 1.5
        
        # ---
        # ROBUST CIRCULARITY: Hardcoded to be forgiving.
        # 0.3 allows for ovals, 1.4 for glare.
        # ---
        self.min_circ = 0.3
        self.max_circ = 1.4
        
        # Get crop settings from config
        self.crop_top = config['video'].get('crop', {}).get('top', 0)
        self.crop_bottom = config['video'].get('crop', {}).get('bottom', 0)
        self.crop_left = config['video'].get('crop', {}).get('left', 0)
        self.crop_right = config['video'].get('crop', {}).get('right', 0)
        
        self.debug = False # Externally set this to True for verbose output

    def set_debug(self, debug_on: bool):
        """Externally set the debug mode."""
        self.debug = debug_on
        
    def detect_balls(self, frame) -> List[Tuple[int, int, int]]:
        """
        Main detection pipeline.
        """
        h, w = frame.shape[:2]
        
        # 1. Crop the frame
        cropped = frame[self.crop_top:h-self.crop_bottom, 
                          self.crop_left:w-self.crop_right]
        
        if cropped.size == 0:
            if self.debug: print("ERROR: Frame was cropped to nothing. Check crop settings.")
            return []
        
        # 2. Convert to HSV
        try:
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        except cv2.error as e:
            print(f"ERROR: cvtColor failed! {e}")
            return []

        # 3. Create Color Mask
        felt_lower = np.array(self.config['ball_detection']['felt_color']['lower'])
        felt_upper = np.array(self.config['ball_detection']['felt_color']['upper'])
        
        mask_felt = cv2.inRange(hsv, felt_lower, felt_upper)
        mask_objects = cv2.bitwise_not(mask_felt)
        
        # ---
        # THE "GOLDEN" MORPHOLOGY
        # 1. A (3, 3) kernel
        # 2. MORPH_OPEN to kill tiny salt noise
        # 3. MORPH_CLOSE (iter=2) to build up broken balls
        # ---
        kernel = np.ones((3, 3), np.uint8)
        mask_opened = cv2.morphologyEx(mask_objects, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 4. Find Contours
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_balls = []

        if self.debug:
            print("\n--- Filtering Contours (Debug Mode) ---")
            print(f"Found {len(contours)} total contours after cleaning.")
            print(f"Filters (DYNAMIC): Area ({self.min_area:.1f}-{self.max_area:.1f}), Radius ({self.min_radius}-{self.max_radius}), Circ ({self.min_circ}-{self.max_circ})")
        
        # 5. Filter Contours
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter 1: Area (Dynamic)
            if not (self.min_area < area < self.max_area):
                if self.debug: print(f"Contour {i}: REJECTED (Area = {area:.2f})")
                continue
            
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # Filter 2: Radius (Dynamic)
            if not (self.min_radius < radius < self.max_radius):
                if self.debug: print(f"Contour {i}: Area OK. REJECTED (Radius = {radius:.2f})")
                continue
            
            # Filter 3: Circularity (Robust)
            circularity = area / (np.pi * (radius ** 2)) if radius > 0 else 0
            if not (self.min_circ < circularity < self.max_circ):
                if self.debug: print(f"Contour {i}: Area/Rad OK. REJECTED (Circularity = {circularity:.2f})")
                continue

            if self.debug: print(f"Contour {i}: PASSED basic filters. Area={area:.2f}, Radius={radius:.2f}, Circ={circularity:.2f}. Verifying...")

            # Convert coords from cropped frame back to original frame
            center_x = int(x) + self.crop_left
            center_y = int(y) + self.crop_top
            radius_int = int(radius)
            
            # Filter 4: Final Verification (Robust)
            if self._verify_ball(frame, center_x, center_y, radius_int):
                if self.debug: print(f"Contour {i}: -> ✅ ACCEPTED")
                detected_balls.append((center_x, center_y, radius_int))
            else:
                if self.debug: print(f"Contour {i}: -> REJECTED (Failed _verify_ball check)")
        
        if self.debug: print("--- Filtering Complete ---")
        
        return detected_balls
    
    def _verify_ball(self, frame, x, y, r) -> bool:
        """
        RELAXED verification. We only reject obvious non-balls.
        """
        h, w = frame.shape[:2]
        
        if x - r < 5 or x + r >= w - 5 or y - r < 5 or y + r >= h - 5:
            return False 
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (x, y), max(1, r - 1), 255, -1)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mean_hsv = cv2.mean(hsv, mask=mask)[:3]
        brightness = mean_hsv[2]
        
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