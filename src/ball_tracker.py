# src/ball_tracker.py
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict, deque
from typing import List, Tuple, Dict


class BallTracker:
    """
    Keeps track of balls over time to assign them persistent IDs.
    """
    
    def __init__(self, config):
        # We start counting IDs at 0.
        self.config = config
        self.next_object_id = 0
        # Dictionary to store the balls. Key = ID, Value = (Location, Radius).
        self.objects = OrderedDict()
        # Count how many frames a ball has been missing. 
        # If it's missing too long, we assume it went in a pocket.
        self.disappeared = OrderedDict()
        # Keep a history of past positions to calculate speed.
        self.position_history = OrderedDict()
        self.max_history = 10
        # Settings from config.yaml
        self.max_disappeared = config['tracking']['max_disappeared']
        self.max_distance = config['tracking']['max_distance']
        self.min_velocity = config['tracking']['min_velocity']
        self.fast_moving = OrderedDict()
        # Pocket settings
        self.pocket_radius = config['pockets']['radius']
        self.pocket_centers = [tuple(loc) for loc in config['pockets']['locations']]

    def _is_in_pocket(self, x: float, y: float) -> bool:
        """Check if x,y is inside a pocket circle."""
        for (px, py) in self.pocket_centers:
            # Pythagorean theorem distance check.
            if np.sqrt((x - px)**2 + (y - py)**2) < self.pocket_radius:
                return True
        return False
        
    def register(self, centroid, radius):
        """Register a new object to track."""
        self.objects[self.next_object_id] = (centroid, radius)
        self.disappeared[self.next_object_id] = 0
        self.position_history[self.next_object_id] = deque(maxlen=self.max_history)
        self.fast_moving[self.next_object_id] = False
        self.next_object_id += 1
        
    def deregister(self, object_id):
        """Remove an object from tracking."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.position_history:
            del self.position_history[object_id]
        if object_id in self.fast_moving:
            del self.fast_moving[object_id]
            
    def update(self, detections: List[Tuple[int, int, int]], frame_number: int):
        """
        The core logic: Match new detections to existing balls.
        """
        # CASE 1: No balls found in this frame.
        if len(detections) == 0:
            # Mark all existing balls as "disappeared".
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                # If gone for too long, remove (deregister) them.
                patience = self.max_disappeared * 2 if self.fast_moving.get(object_id, False) else self.max_disappeared
                if self.disappeared[object_id] > patience:
                    self.deregister(object_id)
            return self.get_tracked_objects(frame_number)

        # Convert the new detections into a math-friendly format (NumPy array).
        input_centroids = np.array([(x, y) for x, y, r in detections])
        input_radii = np.array([r for x, y, r in detections])

        # CASE 2: We are not tracking anything yet.
        if len(self.objects) == 0:
            # Just register everything we see as a new ball.
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_radii[i])
        # CASE 3: We have existing balls and new detections. MATCH THEM!
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array([self.objects[oid][0] for oid in object_ids])
            
            # Calculate the distance between EVERY existing ball and EVERY new detection.
            D = dist.cdist(object_centroids, input_centroids)

            # Find the smallest distances (closest matches).
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()

            # Loop through the matches.
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                # Update the existing ball with the new location.
                object_id = object_ids[row]
                threshold = self.max_distance * 1.5 if self.fast_moving.get(object_id, False) else self.max_distance

                # Check: Is the distance too far? (Did the ball teleport?)
                if D[row, col] > threshold:
                    continue
                    
                self.objects[object_id] = (input_centroids[col], input_radii[col])
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            # If an existing ball was NOT matched, mark it as disappeared.
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                patience = self.max_disappeared * 2 if self.fast_moving.get(object_id, False) else self.max_disappeared
                if self.disappeared[object_id] > patience:
                    self.deregister(object_id)

            # If a new detection was NOT matched, it's a new ball. Register it.
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], input_radii[col])
        
        # Check if any ball fell into a pocket.
        pocketed_ids = []
        for object_id, (centroid, radius) in self.objects.items():
            if self._is_in_pocket(centroid[0], centroid[1]):
                pocketed_ids.append(object_id)
        
        for object_id in pocketed_ids:
            # Remove pocketed balls.
            self.deregister(object_id)
        
            
        # Save position to history (for calculating speed later).
        for object_id in self.objects.keys():
             self.position_history[object_id].append((self.objects[object_id][0][0], self.objects[object_id][0][1], frame_number))
        
        return self.get_tracked_objects(frame_number)
    
    def get_tracked_objects(self, frame_number: int) -> Dict:
        """Get all tracked objects with velocities."""
        result = {}
        
        for object_id, (centroid, radius) in self.objects.items():
            velocity = self.calculate_velocity(object_id, frame_number)
            speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
            
            is_fast = speed > (self.min_velocity * 5)
            self.fast_moving[object_id] = is_fast
            
            result[object_id] = {
                'centroid': tuple(centroid),
                'radius': radius,
                'velocity': velocity,
                'speed': speed,
                'is_moving': speed > self.min_velocity,
                'is_fast': is_fast
            }
        
        return result
    
    def calculate_velocity(self, object_id: int, current_frame: int) -> Tuple[float, float]:
        """Calculate velocity using weighted average of recent positions."""
        if object_id not in self.position_history:
            return (0.0, 0.0)
        
        history = self.position_history[object_id]
        
        if len(history) < 2:
            return (0.0, 0.0)
        
        positions = list(history)
        
        if len(positions) >= 3:
            velocities = []
            weights = []
            
            for i in range(len(positions) - 1, max(0, len(positions) - 5), -1):
                x2, y2, f2 = positions[i]
                x1, y1, f1 = positions[i-1]
                
                if f2 == f1: continue
                
                dt = f2 - f1
                vx = (x2 - x1) / dt
                vy = (y2 - y1) / dt
                
                velocities.append((vx, vy))
                weights.append(1.0 / (len(positions) - i))
            
            if velocities:
                total_weight = sum(weights)
                vx = sum(v[0] * w for v, w in zip(velocities, weights)) / total_weight
                vy = sum(v[1] * w for v, w in zip(velocities, weights)) / total_weight
                
                return (vx, vy)
        
        # Fallback
        x2, y2, f2 = positions[-1]
        x1, y1, f1 = positions[-2]
        
        if f2 == f1: return (0.0, 0.0)
        
        dt = f2 - f1
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        
        return (vx, vy)
