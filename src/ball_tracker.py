import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict, deque
from typing import List, Tuple, Dict


class BallTracker:
    """Tracks pool balls across video frames and calculates their velocities."""
    
    def __init__(self, config):
        self.config = config
        self.next_object_id = 0
        self.objects = OrderedDict()  # {id: (centroid, radius)}
        self.disappeared = OrderedDict()  # {id: frame_count}
        
        # Store position history for velocity calculation
        self.position_history = OrderedDict()  # {id: deque([(x, y, frame), ...])}
        self.max_history = 10  # frames to keep in history
        
        self.max_disappeared = config['tracking']['max_disappeared']
        self.max_distance = config['tracking']['max_distance']
        self.min_velocity = config['tracking']['min_velocity']
        
    def register(self, centroid, radius):
        """Register a new object to track."""
        self.objects[self.next_object_id] = (centroid, radius)
        self.disappeared[self.next_object_id] = 0
        self.position_history[self.next_object_id] = deque(maxlen=self.max_history)
        self.next_object_id += 1
        
    def deregister(self, object_id):
        """Remove an object from tracking."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.position_history:
            del self.position_history[object_id]
            
    def update(self, detections: List[Tuple[int, int, int]], frame_number: int):
        """
        Update tracked objects with new detections.
        
        Args:
            detections: List of (x, y, radius) tuples
            frame_number: Current frame number for velocity calculation
            
        Returns:
            Dict of {object_id: {'centroid': (x, y), 'radius': r, 'velocity': (vx, vy)}}
        """
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
                    
            return self.get_tracked_objects(frame_number)
        
        # Convert detections to numpy array
        input_centroids = np.array([(x, y) for x, y, r in detections])
        input_radii = np.array([r for x, y, r in detections])
        
        # If no objects are being tracked, register all detections
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_radii[i])
        else:
            # Get current object IDs and centroids
            object_ids = list(self.objects.keys())
            object_centroids = np.array([self.objects[oid][0] for oid in object_ids])
            
            # Calculate distances between existing objects and new detections
            D = dist.cdist(object_centroids, input_centroids)
            
            # Find the smallest distance for each existing object
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Associate detections with existing objects
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                    
                if D[row, col] > self.max_distance:
                    continue
                    
                object_id = object_ids[row]
                self.objects[object_id] = (input_centroids[col], input_radii[col])
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle unmatched existing objects
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new detections
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], input_radii[col])
        
        # Update position history for all tracked objects
        for object_id, (centroid, radius) in self.objects.items():
            self.position_history[object_id].append((centroid[0], centroid[1], frame_number))
        
        return self.get_tracked_objects(frame_number)
    
    def get_tracked_objects(self, frame_number: int) -> Dict:
        """
        Get all currently tracked objects with their velocities.
        
        Returns:
            Dict of {object_id: {'centroid': (x, y), 'radius': r, 'velocity': (vx, vy), 'speed': speed}}
        """
        result = {}
        
        for object_id, (centroid, radius) in self.objects.items():
            velocity = self.calculate_velocity(object_id, frame_number)
            speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
            
            result[object_id] = {
                'centroid': tuple(centroid),
                'radius': radius,
                'velocity': velocity,
                'speed': speed,
                'is_moving': speed > self.min_velocity
            }
        
        return result
    
    def calculate_velocity(self, object_id: int, current_frame: int) -> Tuple[float, float]:
        """
        Calculate velocity of an object using its position history.
        
        Returns:
            (vx, vy) velocity in pixels per frame
        """
        if object_id not in self.position_history:
            return (0.0, 0.0)
        
        history = self.position_history[object_id]
        
        if len(history) < 2:
            return (0.0, 0.0)
        
        # Use the last few positions to calculate velocity
        # This smooths out noise in detection
        positions = list(history)
        
        if len(positions) >= 3:
            # Use weighted average of recent velocities
            velocities = []
            weights = []
            
            for i in range(len(positions) - 1, max(0, len(positions) - 4), -1):
                x2, y2, f2 = positions[i]
                x1, y1, f1 = positions[i-1]
                
                if f2 == f1:
                    continue
                
                dt = f2 - f1
                vx = (x2 - x1) / dt
                vy = (y2 - y1) / dt
                
                velocities.append((vx, vy))
                weights.append(1.0 / (len(positions) - i))  # Recent positions weighted more
            
            if velocities:
                total_weight = sum(weights)
                vx = sum(v[0] * w for v, w in zip(velocities, weights)) / total_weight
                vy = sum(v[1] * w for v, w in zip(velocities, weights)) / total_weight
                
                return (vx, vy)
        
        # Fallback to simple velocity calculation
        x2, y2, f2 = positions[-1]
        x1, y1, f1 = positions[-2]
        
        if f2 == f1:
            return (0.0, 0.0)
        
        dt = f2 - f1
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        
        return (vx, vy)
