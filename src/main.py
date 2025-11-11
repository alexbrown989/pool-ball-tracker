import cv2
import yaml
import argparse
import os
import numpy as np
from ball_detector import BallDetector
from ball_tracker import BallTracker
from trajectory_predictor import TrajectoryPredictor


class PoolBallTracker:
    """Main class for pool ball tracking application."""
    
    def __init__(self, config_path='config/config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.detector = None
        self.tracker = None
        self.predictor = None
        
    def process_video(self, input_path: str, output_path: str = None, headless: bool = False):
        """
        Process a video file and track pool balls.
        
        Args:
            input_path: Path to input video
            output_path: Path to save output video (optional)
            headless: Run without display (for Codespaces)
        """
        # Open video
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {input_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        print(f"Headless mode: {headless}")
        
        # Initialize components
        self.detector = BallDetector(self.config)
        self.tracker = BallTracker(self.config)
        self.predictor = TrajectoryPredictor(self.config, width, height)
        
        # Setup output video writer if path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")
        
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Detect balls in current frame
                detections = self.detector.detect_balls(frame)
                
                # Update tracker with detections
                tracked_objects = self.tracker.update(detections, frame_number)
                
                # Draw results on frame
                annotated_frame = self.draw_tracking(frame, tracked_objects)
                
                # Show frame only if not headless
                if not headless:
                    cv2.imshow('Pool Ball Tracker', annotated_frame)
                    
                    # Press 'q' to quit, space to pause
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        cv2.waitKey(0)
                
                # Write to output video
                if writer:
                    writer.write(annotated_frame)
                
                frame_number += 1
                
                # Show progress
                if frame_number % 30 == 0:
                    progress = (frame_number / total_frames) * 100
                    print(f"Progress: {frame_number}/{total_frames} frames ({progress:.1f}%)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if not headless:
                cv2.destroyAllWindows()
            
        print(f"\n✅ Processing complete!")
        print(f"Processed {frame_number} frames")
        if output_path:
            print(f"Output saved to: {output_path}")
    
    def draw_tracking(self, frame, tracked_objects):
        """
        Draws a clean, data-rich HUD (Heads-Up Display) that shows
        stats for ALL moving balls.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # --- 1. Draw the Semi-Transparent HUD Panel ---
        # We make it taller to fit all the new stats
        hud_height = 120 
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_height), (0, 0, 0), -1)
        alpha = 0.7 # Make it slightly more opaque for readability
        annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)

        # --- 2. Draw Text on the HUD ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_small = 0.5
        font_scale_large = 0.7
        font_color = (255, 255, 255)
        line_type = 2
        
        # --- 2a. Find all moving balls ---
        moving_balls_data = []
        for obj_id, obj_data in tracked_objects.items():
            if obj_data['is_moving']:
                moving_balls_data.append((obj_id, obj_data))

        # --- 2b. Draw Summary Info (Top Left) ---
        count_text = f"Balls Tracked: {len(tracked_objects)}"
        moving_text = f"Moving: {len(moving_balls_data)}"
        
        cv2.putText(annotated, count_text, (20, 30), font, font_scale_large, font_color, line_type)
        cv2.putText(annotated, moving_text, (20, 60), font, font_scale_large, font_color, line_type)
        
        # --- 2c. Draw "Cool Metrics" for ALL Moving Balls (Right Side) ---
        
        # We start drawing the list of stats at this y-coordinate
        hud_y_start = 30
        line_height = 25 # How much space between each line of text
        
        cv2.putText(annotated, "--- Active Ball Stats ---", (250, hud_y_start), font, font_scale_small, font_color, 1)
        
        if not moving_balls_data:
            cv2.putText(annotated, "All balls stationary", (250, hud_y_start + line_height), 
                        font, font_scale_small, (200, 200, 200), 1)
        
        # Loop through all moving balls and print their stats on the HUD
        for i, (obj_id, obj_data) in enumerate(moving_balls_data):
            
            # Calculate where to draw this line of text
            current_y = hud_y_start + (i + 1) * line_height
            
            # Stop if we're about to draw off the panel
            if current_y > hud_height - 10:
                break
                
            speed = obj_data['speed']
            velocity = obj_data['velocity']
            
            # Create the data-rich string
            stats_text = (
                f"ID {obj_id}:  Spd: {speed:<5.1f} px/f  |  "
                f"Vel: ({velocity[0]:>5.1f}, {velocity[1]:>5.1f})"
            )
            
            # Draw the text
            cv2.putText(annotated, stats_text, (250, current_y), 
                        font, font_scale_small, font_color, 1)

        # --- 3. Pre-calculate all potential collisions ---
        # (This logic is the same as before, and is very efficient)
        all_collisions = {}
        
        for mover_id, mover_data in moving_balls_data:
            first_collision_time = float('inf')
            first_collision_point = None

            for other_id, other_data in tracked_objects.items():
                if mover_id == other_id: continue
                if other_data['is_moving']: continue 
                
                collision_info = self.predictor.predict_ball_collision(
                    mover_data['centroid'], mover_data['velocity'], mover_data['radius'],
                    other_data['centroid'], other_data['velocity'], other_data['radius']
                )
                
                if collision_info:
                    cx, cy, time = collision_info
                    if time < first_collision_time:
                        first_collision_time = time
                        first_collision_point = (int(cx), int(cy))
            
            if first_collision_point:
                all_collisions[mover_id] = (first_collision_point, first_collision_time)

        # --- 4. Draw on Play Area (ONLY Circles, IDs, and Trajectories) ---
        for obj_id, obj_data in tracked_objects.items():
            centroid = obj_data['centroid']
            radius = obj_data['radius']
            is_moving = obj_data['is_moving']
            
            x, y = int(centroid[0]), int(centroid[1])
            viz_radius = max(1, int(radius))
            
            # Draw the ball circle
            color = tuple(self.config['video']['highlight_color']) if is_moving else (0, 255, 255)
            thickness = 3 if is_moving else 2
            cv2.circle(annotated, (x, y), viz_radius, color, thickness)
            
            # Draw the ID
            id_text = f"ID:{obj_id}"
            cv2.putText(annotated, id_text, (x - 20, y - viz_radius - 10),
                        font, font_scale_small, color, 2)
            
            # Draw trajectory for this ball if it's moving
            if is_moving and self.config['video']['show_trajectory_line']:
                
                trajectory = self.predictor.predict_trajectory(
                    obj_data['centroid'], 
                    obj_data['velocity'], 
                    obj_data['radius']
                )
                
                collision_time = float('inf')
                if obj_id in all_collisions:
                    collision_point, collision_time = all_collisions[obj_id]
                    trajectory = trajectory[:int(collision_time)] # Truncate line
                    
                    # Draw a red "X" at the collision point
                    cx, cy = collision_point
                    cv2.circle(annotated, (cx, cy), 15, (0, 0, 255), 2)
                    cv2.line(annotated, (cx-10, cy-10), (cx+10, cy+10), (0, 0, 255), 2)
                    cv2.line(annotated, (cx+10, cy-10), (cx-10, cy+10), (0, 0, 255), 2)

                # Draw the (potentially truncated) trajectory line
                if len(trajectory) > 1:
                    pts = np.array(trajectory, dtype=np.int32)
                    cv2.polylines(annotated, [pts], isClosed=False, 
                                  color=tuple(self.config['video']['trajectory_color']), 
                                  thickness=2)
                
                # If no collision, draw the final "ghost ball" at the cushion
                if collision_time == float('inf') and trajectory:
                    final_pos = trajectory[-1]
                    cv2.circle(annotated, final_pos, viz_radius, 
                               tuple(self.config['video']['trajectory_color']), 
                               2, cv2.LINE_AA)
        
        return annotated


def main():
    parser = argparse.ArgumentParser(description='Pool Ball Tracker')
    parser.add_argument('input', type=str, help='Path to input video file')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Path to output video file (optional)')
    parser.add_argument('-c', '--config', type=str, default='../config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--headless', action='store_true',
                       help='Run without display (for Codespaces/servers)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        return
    
    # Create output directory if needed
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Run tracker
    tracker = PoolBallTracker(args.config)
    tracker.process_video(args.input, args.output, headless=args.headless)


if __name__ == '__main__':
    main()