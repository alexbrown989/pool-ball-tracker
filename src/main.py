import cv2
import yaml
import argparse
import os
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
        """Draw tracking information on the frame."""
        annotated = frame.copy()
        
        highlight_color = tuple(self.config['video']['highlight_color'])
        trajectory_color = tuple(self.config['video']['trajectory_color'])
        
        for obj_id, obj_data in tracked_objects.items():
            centroid = obj_data['centroid']
            radius = obj_data['radius']
            velocity = obj_data['velocity']
            speed = obj_data['speed']
            is_moving = obj_data['is_moving']
            
            x, y = centroid
            
            # Draw ball circle
            if is_moving:
                # Highlight moving balls
                color = highlight_color
                thickness = 3
            else:
                # Draw stationary balls normally
                color = (0, 255, 255)  # Yellow for stationary
                thickness = 2
            
            cv2.circle(annotated, (int(x), int(y)), radius, color, thickness)
            
            # Draw object ID
            text = f"ID:{obj_id}"
            cv2.putText(annotated, text, (int(x) - 20, int(y) - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # If ball is moving, show velocity and trajectory
            if is_moving and self.config['video']['show_velocity_text']:
                # Draw velocity vector
                vx, vy = velocity
                end_x = int(x + vx * 10)  # Scale for visibility
                end_y = int(y + vy * 10)
                cv2.arrowedLine(annotated, (int(x), int(y)), (end_x, end_y), 
                              color, 2, tipLength=0.3)
                
                # Display speed
                speed_text = f"v:{speed:.1f} px/f"
                cv2.putText(annotated, speed_text, (int(x) - 30, int(y) + radius + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Display velocity components
                vel_text = f"({vx:.1f}, {vy:.1f})"
                cv2.putText(annotated, vel_text, (int(x) - 30, int(y) + radius + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Predict and draw trajectory for moving balls
            if is_moving and self.config['video']['show_trajectory_line']:
                trajectory = self.predictor.predict_trajectory(centroid, velocity, radius)
                
                if len(trajectory) > 1:
                    # Draw trajectory line
                    for i in range(len(trajectory) - 1):
                        pt1 = trajectory[i]
                        pt2 = trajectory[i + 1]
                        # Fade the line as it goes further
                        alpha = 1.0 - (i / len(trajectory))
                        color_alpha = tuple(int(c * alpha) for c in trajectory_color)
                        cv2.line(annotated, pt1, pt2, color_alpha, 2)
                    
                    # Draw predicted collision point
                    collision_point = self.predictor.find_collision_point(trajectory)
                    if collision_point:
                        cv2.circle(annotated, collision_point, 8, (0, 0, 255), -1)
                        cv2.putText(annotated, "Impact", 
                                  (collision_point[0] - 25, collision_point[1] - 15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add frame info
        info_text = f"Frame: {len(tracked_objects)} balls tracked"
        cv2.putText(annotated, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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