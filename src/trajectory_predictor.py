import numpy as np
from typing import List, Tuple, Optional


class TrajectoryPredictor:
    """Predicts ball trajectories considering friction and cushion bounces."""
    
    def __init__(self, config, frame_width: int, frame_height: int):
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.friction = config['trajectory']['friction_coefficient']
        self.prediction_frames = config['trajectory']['prediction_frames']
        self.cushion_bounce = config['trajectory']['cushion_bounce']
        
        # Define table boundaries (approximate, based on typical pool table in frame)
        # These might need adjustment based on your specific video
        self.cushion_margin = 50  # pixels from edge
        self.left_cushion = self.cushion_margin
        self.right_cushion = frame_width - self.cushion_margin
        self.top_cushion = self.cushion_margin
        self.bottom_cushion = frame_height - self.cushion_margin
        
    def predict_trajectory(self, position: Tuple[float, float], 
                          velocity: Tuple[float, float],
                          radius: float) -> List[Tuple[int, int]]:
        """
        Predict the trajectory of a ball.
        
        Args:
            position: Current (x, y) position
            velocity: Current (vx, vy) velocity in pixels/frame
            radius: Ball radius
            
        Returns:
            List of (x, y) predicted positions
        """
        trajectory = []
        
        x, y = float(position[0]), float(position[1])
        vx, vy = float(velocity[0]), float(velocity[1])
        
        for frame in range(self.prediction_frames):
            # Apply friction (velocity decay)
            vx *= self.friction
            vy *= self.friction
            
            # Stop if velocity becomes negligible
            speed = np.sqrt(vx**2 + vy**2)
            if speed < 0.1:
                break
            
            # Update position
            x += vx
            y += vy
            
            # Check for cushion collisions and bounce
            bounced = False
            
            # Left/Right cushion
            if x - radius <= self.left_cushion:
                x = self.left_cushion + radius
                vx = -vx * self.cushion_bounce
                bounced = True
            elif x + radius >= self.right_cushion:
                x = self.right_cushion - radius
                vx = -vx * self.cushion_bounce
                bounced = True
            
            # Top/Bottom cushion
            if y - radius <= self.top_cushion:
                y = self.top_cushion + radius
                vy = -vy * self.cushion_bounce
                bounced = True
            elif y + radius >= self.bottom_cushion:
                y = self.bottom_cushion - radius
                vy = -vy * self.cushion_bounce
                bounced = True
            
            # Add position to trajectory
            if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                trajectory.append((int(x), int(y)))
            else:
                break  # Ball went off screen
        
        return trajectory
    
    def find_collision_point(self, trajectory: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Find where the ball will hit a cushion.
        
        Returns:
            (x, y) of collision point, or None if no collision predicted
        """
        for i, (x, y) in enumerate(trajectory):
            # Check if near a cushion
            if (abs(x - self.left_cushion) < 10 or 
                abs(x - self.right_cushion) < 10 or
                abs(y - self.top_cushion) < 10 or 
                abs(y - self.bottom_cushion) < 10):
                return (x, y)
        
        # Return the end of trajectory if no cushion hit
        if trajectory:
            return trajectory[-1]
        
        return None
    
    def estimate_time_to_collision(self, trajectory: List[Tuple[int, int]]) -> Optional[float]:
        """
        Estimate time (in frames) until collision.
        
        Returns:
            Number of frames until collision, or None
        """
        collision_point = self.find_collision_point(trajectory)
        
        if collision_point and trajectory:
            # Find index of collision point in trajectory
            try:
                idx = trajectory.index(collision_point)
                return float(idx)
            except ValueError:
                return None
        
        return None
    
    def predict_ball_collision(self, ball1_pos: Tuple[float, float],
                              ball1_vel: Tuple[float, float],
                              ball1_radius: float,
                              ball2_pos: Tuple[float, float],
                              ball2_vel: Tuple[float, float],
                              ball2_radius: float) -> Optional[Tuple[int, int, float]]:
        """
        Predict if and where two balls will collide.
        
        Returns:
            (x, y, time_in_frames) of collision, or None if no collision predicted
        """
        # Simplified collision prediction
        # For a more accurate prediction, you'd need to solve the quadratic equation
        # for the intersection of two moving circles
        
        x1, y1 = ball1_pos
        vx1, vy1 = ball1_vel
        x2, y2 = ball2_pos
        vx2, vy2 = ball2_vel
        
        min_dist = ball1_radius + ball2_radius
        
        for t in range(self.prediction_frames):
            # Predict positions at time t
            pred_x1 = x1 + vx1 * t * (self.friction ** t)
            pred_y1 = y1 + vy1 * t * (self.friction ** t)
            pred_x2 = x2 + vx2 * t * (self.friction ** t)
            pred_y2 = y2 + vy2 * t * (self.friction ** t)
            
            # Check distance
            dist = np.sqrt((pred_x1 - pred_x2)**2 + (pred_y1 - pred_y2)**2)
            
            if dist <= min_dist:
                # Collision detected
                collision_x = int((pred_x1 + pred_x2) / 2)
                collision_y = int((pred_y1 + pred_y2) / 2)
                return (collision_x, collision_y, float(t))
        
        return None
