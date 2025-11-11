import cv2
import yaml
import sys
import argparse
import os

# Add the root directory to the Python path
sys.path.insert(0, '.')
from ball_detector import BallDetector # This will now work!

def visualize_detections(config_path: str, video_path: str, output_path: str, frame_num: int, crop_test_only: bool = False):
    """
    Loads a specific frame from a video.
    If crop_test_only is True, it draws the crop box and saves.
    If False, it runs detection and saves the visualization.
    """
    
    # --- 1. Load Config ---
    if not os.path.exists(config_path):
        print(f"FATAL ERROR: Config file not found at: {config_path}")
        return
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from {config_path}")

    # --- 2. Load Video and Seek to Frame ---
    if not os.path.exists(video_path):
        print(f"FATAL ERROR: Video file not found at: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"FATAL ERROR: Could not open video file: {video_path}")
        return

    # Seek to the desired frame
    if frame_num > 0:
        print(f"Seeking to frame {frame_num}...")
        for _ in range(frame_num): # Use _ for unused loop variable
            ret, _ = cap.read()
            if not ret:
                print(f"ERROR: Video ended before reaching frame {frame_num}.")
                cap.release()
                return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("FATAL ERROR: Could not read target frame.")
        return
        
    h, w = frame.shape[:2]
    print(f"Successfully loaded frame {frame_num} (Resolution: {w}x{h})")

    # --- 3. CROP TEST MODE ---
    if crop_test_only:
        print("\n--- RUNNING IN CROP TEST MODE ---")
        crop_config = config.get('video', {}).get('crop', {})
        top = crop_config.get('top', 0)
        bottom = crop_config.get('bottom', 0)
        left = crop_config.get('left', 0)
        right = crop_config.get('right', 0)
        
        print(f"Loaded crop values: top={top}, bottom={bottom}, left={left}, right={right}")

        # Define the crop boundaries
        y1 = top
        y2 = h - bottom
        x1 = left
        x2 = w - right

        output = frame.copy()
        
        # Draw a thick, bright white rectangle showing the *active* detection area
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 255), 3)
        
        # Save and report
        crop_output_path = "../output/test_crop.jpg" # Save to a dedicated file
        try:
            cv2.imwrite(crop_output_path, output)
            print(f"\n✅ Saved crop test image to: {crop_output_path}")
            print("---")
            print("NEXT STEP: Open that image. Is the white box *perfectly* around the felt?")
            print("If NOT, edit config.yaml, save, and run this script again.")
            print("If YES, edit this script, set CROP_TEST = False, and run again.")
            print("---")
        except Exception as e:
            print(f"FATAL ERROR: Could not save crop test image. {e}")
        return # We are done

    # --- 4. Run Detection (if not in crop test mode) ---
    detector = BallDetector(config)
    detector.set_debug(True)  # Set debug mode properly

    print("\nDetecting balls...")
    detections = detector.detect_balls(frame)
    print(f"\n✅ FOUND {len(detections)} BALLS!")

    # --- 5. Draw VISIBLE Results ---
    output = frame.copy()
    VIZ_RADIUS = 20  # Draw a 20-pixel circle
    VIZ_THICKNESS = 2
    
    for (x, y, r) in detections:
        cv2.circle(output, (x, y), VIZ_RADIUS, (0, 255, 0), VIZ_THICKNESS)
        cv2.circle(output, (x, y), 2, (0, 0, 255), -1)
        text = f"Ball @ ({x},{y}) | Detected r={r}"
        cv2.putText(output, text, (x - 70, y - VIZ_RADIUS - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # --- 6. Save and Report ---
    try:
        cv2.imwrite(output_path, output)
        print(f"\nSaved visualization to: {output_path}")
    except Exception as e:
        print(f"FATAL ERROR: Could not save output image. {e}")

    print("\n--- Detection Summary ---")
    for i, (x, y, r) in enumerate(detections):
        print(f"Ball {i+1}: center=({x}, {y}), detected_radius={r}")

if __name__ == "__main__":
    
    # ---
    # --- MANUAL SWITCH ---
    # ---
    # SET THIS TO TRUE to find your crop values
    # SET THIS TO FALSE to run detection
    CROP_TEST = False
    # ---
    
    parser = argparse.ArgumentParser(description="Test and visualize ball detection on a single frame.")
    
    parser.add_argument(
        '-c', '--config', 
        default='../config/config.yaml', 
        help="Path to the config.yaml file"
    )
    parser.add_argument(
        '-i', '--input', 
        default='../data/videos/pool_fixed.mp4', 
        help="Path to the input video file"
    )
    parser.add_argument(
        '-o', '--output', 
        default='../output/test_detection.jpg', 
        help="Path to save the output visualization"
    )
    parser.add_argument(
        '-f', '--frame', 
        type=int, 
        default=0, 
        help="Frame number to test (default: 0, the first frame)"
    )
    
    args = parser.parse_args()
    
    visualize_detections(args.config, args.input, args.output, args.frame, crop_test_only=CROP_TEST)