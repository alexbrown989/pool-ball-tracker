
# src/trajectory_predictor.pyimport numpy as np
from typing import List, Tuple, Optional

class TrajectoryPredictor:
    """
    Predicts where balls will go, bouncing off cushions.
    """
    
    def __init__(self, config, frame_width: int, frame_height: int):
        # Physics settings from config
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.friction = config['trajectory']['friction_coefficient']
        self.prediction_frames = config['trajectory']['prediction_frames']
        self.cushion_bounce = config['trajectory']['cushion_bounce']
        
        # Define where the cushions (walls) are.
        crop_cfg = config['video'].get('crop', {})
        self.left_cushion = crop_cfg.get('left', 10)
        self.right_cushion = frame_width - crop_cfg.get('right', 10)
        self.top_cushion = crop_cfg.get('top', 10)
        self.bottom_cushion = frame_height - crop_cfg.get('bottom', 10)
        
        # --- NEW POCKET DEFINITIONS ---
        self.pocket_radius = config['pockets']['radius']
        self.pocket_centers = [tuple(loc) for loc in config['pockets']['locations']]

    def _is_in_pocket(self, x: float, y: float) -> bool:
        """Helper function to check if a point (x, y) is inside any pocket."""
        for (px, py) in self.pocket_centers:
            if np.sqrt((x - px)**2 + (y - py)**2) < self.pocket_radius:
                return True
        return False

    def predict_trajectory(self, position: Tuple[float, float], 
                           velocity: Tuple[float, float],
                           radius: float) -> List[Tuple[int, int]]:
        """
        Simulate the future!
        Returns a list of points: [(x, y), (x, y), ...] representing the path.
        """
        trajectory = []

        # Start at current position/speed.
        x, y = float(position[0]), float(position[1])
        vx, vy = float(velocity[0]), float(velocity[1])

        # Loop into the future (e.g., next 30 frames).
        for frame in range(self.prediction_frames):
            
            # 1. Apply Friction: Slow the ball down slightly every frame.
            vx *= self.friction
            vy *= self.friction
            
            # If ball is barely moving, stop predicting.
            speed = np.sqrt(vx**2 + vy**2)
            if speed < 0.1:
                break
            
            # 2. Move the ball.
            x += vx
            y += vy
            
            # --- THIS IS THE NEW FIX ---
            # Check if the ball's center has entered a pocket
            if self._is_in_pocket(x, y):
                trajectory.append((int(x), int(y))) # Add the final point
                break # Stop predicting, the ball is pocketed
            

            # 3. Check for cushion hits (Bouncing).
            # If hit Left Wall:
            if x - radius <= self.left_cushion:
                x = self.left_cushion + radius # Reset position so it doesn't get stuck
                vx = -vx * self.cushion_bounce # Reverse X direction (Bounce!)
            # If hit Right Wall:
            elif x + radius >= self.right_cushion:
                x = self.right_cushion - radius
                vx = -vx * self.cushion_bounce

            # If hit Top Wall:
            if y - radius <= self.top_cushion:
                y = self.top_cushion + radius
                vy = -vy * self.cushion_bounce
            # If hit Bottom Wall:
            elif y + radius >= self.bottom_cushion:
                y = self.bottom_cushion - radius
                vy = -vy * self.cushion_bounce

            # Add this future point to our list.
            trajectory.append((int(x), int(y)))
        
        return trajectory
    
    # ... (previous code for __init__ and predict_trajectory) ...
    def find_collision_point(self, trajectory: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        # Loop through every point in the predicted path.
        for i, (x, y) in enumerate(trajectory):
            # Check proximity: Are we within 10 pixels of any wall?
            # We check Left, Right, Top, and Bottom.
            if (abs(x - self.left_cushion) < 10 or 
                abs(x - self.right_cushion) < 10 or
                abs(y - self.top_cushion) < 10 or 
                abs(y - self.bottom_cushion) < 10):
                # If yes, we found the impact point! Return it.
                return (x, y)

        # If we checked the whole path and hit nothing, 
        # just return the very last point where the ball stops.
        if trajectory: return trajectory[-1]
        return None
        
    def estimate_time_to_collision(self, trajectory: List[Tuple[int, int]]) -> Optional[float]:
        """
        Figure out WHEN the collision happens (in frame numbers).
        """
        # First, find WHERE the collision is using the function above.
        collision_point = self.find_collision_point(trajectory)
        if collision_point and trajectory:
            try:
                # Find the 'index' of that point in the list.
                # Since the list is ordered by time (frame 1, frame 2...),
                # the index IS the time (e.g., index 5 = 5 frames from now).
                idx = trajectory.index(collision_point)
                return float(idx)
            except ValueError: return None
        return None
        
    def predict_ball_collision(self, ball1_pos, ball1_vel, ball1_radius,
                               ball2_pos, ball2_vel, ball2_radius) -> Optional[Tuple[int, int, float]]:
        """
        Advanced Math: Will two balls crash into each other?
        We simulate both moving forward to see if they touch.
        """
        # Unpack the current position and speed of Ball 1
        x1, y1 = ball1_pos; vx1, vy1 = ball1_vel
        # Unpack the current position and speed of Ball 2
        x2, y2 = ball2_pos; vx2, vy2 = ball2_vel
        # Calculate the "Touch Distance". 
        # If the centers of the balls are closer than (r1 + r2), they are touching.
        min_dist = ball1_radius + ball2_radius

        # Simulate the future, frame by frame.
        # (friction ** t) means friction * friction * friction... t times.
        # This makes the balls slow down as 't' gets bigger.
        for t in range(self.prediction_frames):
            # Predict Ball 1's future position at time 't'
            pred_x1 = x1 + vx1 * t * (self.friction ** t)
            pred_y1 = y1 + vy1 * t * (self.friction ** t)
            # Predict Ball 2's future position at time 't'
            pred_x2 = x2 + vx2 * t * (self.friction ** t)
            pred_y2 = y2 + vy2 * t * (self.friction ** t)
            # Calculate distance between the two predicted future points.
            # (Standard Euclidean distance formula).
            dist = np.sqrt((pred_x1 - pred_x2)**2 + (pred_y1 - pred_y2)**2)

            # CHECK: Are they touching?
            if dist <= min_dist:
                # Yes! Collision detected.
                
                # Find the middle point between them (where the crash happens).
                collision_x = int((pred_x1 + pred_x2) / 2)
                collision_y = int((pred_y1 + pred_y2) / 2)
                # Return WHERE (x, y) and WHEN (t) it happens.
                return (collision_x, collision_y, float(t))
        # If we finished the loop and they never touched, return None.
        return None
