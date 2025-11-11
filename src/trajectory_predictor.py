import numpy as np
from typing import List, Tuple, Optional

class TrajectoryPredictor:
    """Predicts ball trajectories, now with pocket awareness."""
    
    def __init__(self, config, frame_width: int, frame_height: int):
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.friction = config['trajectory']['friction_coefficient']
        self.prediction_frames = config['trajectory']['prediction_frames']
        self.cushion_bounce = config['trajectory']['cushion_bounce']
        
        # --- DYNAMIC CUSHION FIX ---
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
            # Check distance from point to pocket center
            if np.sqrt((x - px)**2 + (y - py)**2) < self.pocket_radius:
                return True
        return False

    def predict_trajectory(self, position: Tuple[float, float], 
                           velocity: Tuple[float, float],
                           radius: float) -> List[Tuple[int, int]]:
        """
        Predict the trajectory of a ball, stopping at cushions OR pockets.
        """
        trajectory = []
        
        x, y = float(position[0]), float(position[1])
        vx, vy = float(velocity[0]), float(velocity[1])
        
        for frame in range(self.prediction_frames):
            # Apply friction
            vx *= self.friction
            vy *= self.friction
            
            # Stop if velocity is negligible
            speed = np.sqrt(vx**2 + vy**2)
            if speed < 0.1:
                break
            
            # Update position
            x += vx
            y += vy
            
            # --- THIS IS THE NEW FIX ---
            # Check if the ball's center has entered a pocket
            if self._is_in_pocket(x, y):
                trajectory.append((int(x), int(y))) # Add the final point
                break # Stop predicting, the ball is pocketed
            # --- END OF FIX ---

            # Check for cushion collisions and bounce
            if x - radius <= self.left_cushion:
                x = self.left_cushion + radius
                vx = -vx * self.cushion_bounce
            elif x + radius >= self.right_cushion:
                x = self.right_cushion - radius
                vx = -vx * self.cushion_bounce
            
            if y - radius <= self.top_cushion:
                y = self.top_cushion + radius
                vy = -vy * self.cushion_bounce
            elif y + radius >= self.bottom_cushion:
                y = self.bottom_cushion - radius
                vy = -vy * self.cushion_bounce
            
            trajectory.append((int(x), int(y)))
        
        return trajectory
    
    # ... (rest of your functions, find_collision_point, etc., are fine) ...
    def find_collision_point(self, trajectory: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        for i, (x, y) in enumerate(trajectory):
            if (abs(x - self.left_cushion) < 10 or 
                abs(x - self.right_cushion) < 10 or
                abs(y - self.top_cushion) < 10 or 
                abs(y - self.bottom_cushion) < 10):
                return (x, y)
        if trajectory: return trajectory[-1]
        return None
        
    def estimate_time_to_collision(self, trajectory: List[Tuple[int, int]]) -> Optional[float]:
        collision_point = self.find_collision_point(trajectory)
        if collision_point and trajectory:
            try:
                idx = trajectory.index(collision_point)
                return float(idx)
            except ValueError: return None
        return None
        
    def predict_ball_collision(self, ball1_pos, ball1_vel, ball1_radius,
                               ball2_pos, ball2_vel, ball2_radius) -> Optional[Tuple[int, int, float]]:
        x1, y1 = ball1_pos; vx1, vy1 = ball1_vel
        x2, y2 = ball2_pos; vx2, vy2 = ball2_vel
        min_dist = ball1_radius + ball2_radius
        
        for t in range(self.prediction_frames):
            pred_x1 = x1 + vx1 * t * (self.friction ** t)
            pred_y1 = y1 + vy1 * t * (self.friction ** t)
            pred_x2 = x2 + vx2 * t * (self.friction ** t)
            pred_y2 = y2 + vy2 * t * (self.friction ** t)
            dist = np.sqrt((pred_x1 - pred_x2)**2 + (pred_y1 - pred_y2)**2)
            
            if dist <= min_dist:
                collision_x = int((pred_x1 + pred_x2) / 2)
                collision_y = int((pred_y1 + pred_y2) / 2)
                return (collision_x, collision_y, float(t))
        return None