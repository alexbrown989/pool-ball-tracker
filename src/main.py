import cv2              # "OpenCV": The main tool for processing images and video.
import yaml             # A tool to read our settings file (config.yaml).
import argparse         # A tool to read commands you type in the terminal.
import os               # A tool to work with files and folders on your computer.
import numpy as np      # "NumPy": A math tool for working with lists of numbers.

# These are the custom tools we wrote in the other files.
from ball_detector import BallDetector
from ball_tracker import BallTracker
from trajectory_predictor import TrajectoryPredictor


class PoolBallTracker:
    
    """
    This is the main 'Manager' class. 
    It coordinates the Detector (Eyes), Tracker (Memory), and Predictor (Fortune Teller).
    """
    
    def __init__(self, config_path='config/config.yaml'):
        # This function runs ONCE when you start the program.
        
        # Open the configuration file to read our settings.
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Create placeholders for our tools. We will set them up later.
        self.detector = None
        self.tracker = None
        self.predictor = None
        
    def process_video(self, input_path: str, output_path: str = None, headless: bool = False):
        """
        The main loop. It goes through the video one picture (frame) at a time.
        """
        # specialized tool from OpenCV to open the video file.
        cap = cv2.VideoCapture(input_path)

        # Check if the video actually opened. If not, stop and complain.
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_path}")
            return
        
        # Get information about the video (like how fast it plays and how big it is).
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Print a summary to the terminal so we know it's working.
        print(f"Processing video: {input_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        print(f"Headless mode: {headless}")
        
        # --- INITIALIZE OUR TOOLS ---
        # 1. The Detector needs the config to know what colors to look for.
        self.detector = BallDetector(self.config)
        # 2. The Tracker needs settings to know how fast balls move.
        self.tracker = BallTracker(self.config)
        # 3. The Predictor needs the screen size to know where the walls are.
        self.predictor = TrajectoryPredictor(self.config, width, height)
        
        # If the user wants to save the result, set up a "Video Writer".
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")
        
        frame_number = 0
        
        # --- THE MAIN LOOP ---
        try:
            while True:
                # 1. Read the next picture (frame) from the video.
                # 'ret' tells us if it worked, 'frame' is the actual picture.
                
                ret, frame = cap.read()

                # If 'ret' is False, the video is over. Stop the loop.
                if not ret:
                    break
                
                # 2. ASK THE DETECTOR: "Where are the balls in this picture?"
                detections = self.detector.detect_balls(frame)
                
                # 3. ASK THE TRACKER: "Which ball is which?"
                # It matches the new locations to the old ones to give them IDs.
                tracked_objects = self.tracker.update(detections, frame_number)
                
                # 4. DRAW EVERYTHING: Draw circles, lines, and text on the picture.
                annotated_frame = self.draw_tracking(frame, tracked_objects)
                
                # 5. Show the picture on the screen (unless we are in 'headless' mode).
                if not headless:
                    cv2.imshow('Pool Ball Tracker', annotated_frame)
                    
                    # Wait 1 millisecond for a key press.
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): # If user presses 'q', quit.
                        break
                    elif key == ord(' '): # If user presses space, pause.
                        cv2.waitKey(0)
                
                # 6. Save the picture to the output video file (if configured).
                if writer:
                    writer.write(annotated_frame)
                
                frame_number += 1
                
                # Print progress every 30 frames so we know it hasn't frozen.
                if frame_number % 30 == 0:
                    progress = (frame_number / total_frames) * 100
                    print(f"Progress: {frame_number}/{total_frames} frames ({progress:.1f}%)")

        # This runs when the loop finishes (or crashes).
        finally:
            # Clean up: Close the video files and windows.
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
        This function handles the "art" - drawing the HUD and overlays.
        """
        # Make a copy of the frame so we don't draw on the original data.
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # --- DRAW THE BLACK BOX AT THE TOP (HUD) ---
        hud_height = 120 
        overlay = annotated.copy()
        # Draw a black rectangle at the top.
        cv2.rectangle(overlay, (0, 0), (w, hud_height), (0, 0, 0), -1)
        # Blend it with the original to make it see-through (transparent).
        alpha = 0.7 
        annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)

        # Settings for the text (font, size, color).
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_small = 0.5
        font_scale_large = 0.7
        font_color = (255, 255, 255)
        line_type = 2
        
        # Filter the list to find only balls that are moving.
        moving_balls_data = []
        for obj_id, obj_data in tracked_objects.items():
            if obj_data['is_moving']:
                moving_balls_data.append((obj_id, obj_data))

        # --- Summary Info (Top Left) ---
        count_text = f"Balls Tracked: {len(tracked_objects)}"
        moving_text = f"Moving: {len(moving_balls_data)}"
        cv2.putText(annotated, count_text, (20, 30), font, font_scale_large, font_color, line_type)
        cv2.putText(annotated, moving_text, (20, 60), font, font_scale_large, font_color, line_type)
        
        # --- DRAW TEXT FOR EACH MOVING BALL ---
        hud_y_start = 30
        line_height = 25 
        
        cv2.putText(annotated, "--- Active Ball Stats ---", (250, hud_y_start), font, font_scale_small, font_color, 1)
        
        if not moving_balls_data:
            cv2.putText(annotated, "All balls stationary", (250, hud_y_start + line_height), 
                        font, font_scale_small, (200, 200, 200), 1)
        
        # Loop through moving balls and print their speed.
        for i, (obj_id, obj_data) in enumerate(moving_balls_data):
            current_y = hud_y_start + (i + 1) * line_height
            # Don't draw off the screen.
            if current_y > hud_height - 10:
                break
                
            speed = obj_data['speed']
            velocity = obj_data['velocity']
            # --- DRAW CIRCLES AND LINES ON THE TABLE ---
            stats_text = (
                f"ID {obj_id}:  Spd: {speed:<5.1f} px/f  |  "
                f"Vel: ({velocity[0]:>5.1f}, {velocity[1]:>5.1f})"
            )
            cv2.putText(annotated, stats_text, (250, current_y), 
                        font, font_scale_small, font_color, 1)

        # --- 3. Pre-calculate all potential collisions ---
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

        # --- DRAW CIRCLES AND LINES ON THE TABLE ---
        for obj_id, obj_data in tracked_objects.items():
            # Get position and radius.
            centroid = obj_data['centroid']
            radius = obj_data['radius']
            is_moving = obj_data['is_moving']
            x, y = int(centroid[0]), int(centroid[1])
            viz_radius = max(1, int(radius))
            
            # Choose color: Green if moving, Yellow if stopped.
            color = tuple(self.config['video']['highlight_color']) if is_moving else (0, 255, 255)
            thickness = 3 if is_moving else 2
            # Draw the circle around the ball.
            cv2.circle(annotated, (x, y), viz_radius, color, thickness)
            
            # Draw the ID number next to it.
            id_text = f"ID:{obj_id}"
            cv2.putText(annotated, id_text, (x - 20, y - viz_radius - 10),
                        font, font_scale_small, color, 2)
            
            # If moving, ASK THE PREDICTOR: "Where is this going?"
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

                # Draw the predicted path as a line.
                if len(trajectory) > 1:
                    pts = np.array(trajectory, dtype=np.int32)
                    # 'polylines' connects the dots to form a line.
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

# This ensures the program starts when you run 'python src/main.py'
if __name__ == '__main__':
    main()
