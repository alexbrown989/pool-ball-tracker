# Pool Ball Tracker - Methods and Functionality Explanation

This document provides a comprehensive explanation of how every component and method in the Pool Ball Tracker system functions.

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Main Module (`main.py`)](#main-module-mainpy)
3. [Ball Detector Module (`ball_detector.py`)](#ball-detector-module-ball_detectorpy)
4. [Ball Tracker Module (`ball_tracker.py`)](#ball-tracker-module-ball_trackerpy)
5. [Trajectory Predictor Module (`trajectory_predictor.py`)](#trajectory-predictor-module-trajectory_predictorpy)
6. [Test Detection Module (`test_detection.py`)](#test-detection-module-test_detectionpy)
7. [Configuration System](#configuration-system)

---

## System Architecture Overview

The Pool Ball Tracker is a computer vision system that processes video frames to:
- **Detect** pool balls using color-based image processing
- **Track** balls across frames with persistent IDs
- **Predict** future trajectories using physics simulation
- **Visualize** results with overlays and HUD information

The system follows a pipeline architecture:
```
Video Frame → Detection → Tracking → Prediction → Visualization → Output
```

---

## Main Module (`main.py`)

The main module orchestrates the entire tracking pipeline.

### Class: `PoolBallTracker`

#### `__init__(self, config_path='config/config.yaml')`

**Purpose**: Initializes the tracker system by loading configuration and preparing component placeholders.

**How it works**:
1. Opens and parses the YAML configuration file using `yaml.safe_load()`
2. Stores the configuration in `self.config` for access by all components
3. Initializes three component placeholders as `None`:
   - `self.detector`: Will hold the `BallDetector` instance
   - `self.tracker`: Will hold the `BallTracker` instance
   - `self.predictor`: Will hold the `TrajectoryPredictor` instance

**Why placeholders**: Components are created later in `process_video()` after video properties (width, height) are known, which are needed for proper initialization.

---

#### `process_video(self, input_path: str, output_path: str = None, headless: bool = False)`

**Purpose**: The main processing loop that handles video frame-by-frame processing.

**How it works**:

1. **Video Initialization**:
   - Opens video file using `cv2.VideoCapture()`
   - Extracts metadata: FPS, width, height, total frame count
   - Validates video opened successfully

2. **Component Initialization**:
   - Creates `BallDetector` with config (needs color thresholds, radius limits)
   - Creates `BallTracker` with config (needs tracking parameters)
   - Creates `TrajectoryPredictor` with config + video dimensions (needs screen boundaries)

3. **Output Setup** (if `output_path` provided):
   - Creates `cv2.VideoWriter` with MP4V codec
   - Matches input video's FPS and resolution

4. **Main Processing Loop**:
   ```
   For each frame:
     a. Read frame from video
     b. Detect balls → get list of (x, y, radius) tuples
     c. Update tracker → match detections to existing IDs
     d. Draw visualization → overlay circles, lines, text
     e. Display frame (unless headless mode)
     f. Write frame to output video (if configured)
     g. Handle user input (q=quit, space=pause)
   ```

5. **Cleanup**:
   - Releases video capture and writer resources
   - Closes OpenCV windows
   - Prints completion statistics

**Key Design Decisions**:
- Progress reporting every 30 frames to avoid console spam
- Try-finally block ensures cleanup even on errors
- Headless mode allows running on servers without display

---

#### `draw_tracking(self, frame, tracked_objects)`

**Purpose**: Creates the visual overlay (HUD and annotations) on each frame.

**How it works**:

1. **Frame Copy**: Creates a copy to avoid modifying original data

2. **HUD Background**:
   - Draws semi-transparent black rectangle at top (120px height)
   - Uses `cv2.addWeighted()` for transparency effect (alpha=0.7)

3. **HUD Statistics** (Top-left):
   - Total balls tracked count
   - Number of moving balls
   - Uses `cv2.putText()` with white color

4. **Active Ball Stats** (Top-right):
   - Filters `tracked_objects` to find only moving balls
   - For each moving ball, displays:
     - ID number
     - Speed (pixels per frame)
     - Velocity vector (vx, vy)

5. **Collision Prediction**:
   - Pre-calculates all potential ball-to-ball collisions
   - For each moving ball, checks collision with all stationary balls
   - Uses `predictor.predict_ball_collision()` to find collision point and time
   - Stores earliest collision for each ball

6. **On-Table Visualization**:
   - **Circles**: Draws around each detected ball
     - Green for moving balls (thickness=3)
     - Yellow for stationary balls (thickness=2)
   - **ID Labels**: Text showing "ID:X" above each ball
   - **Trajectory Lines**: For moving balls:
     - Calls `predictor.predict_trajectory()` to get future path
     - Truncates path if collision detected
     - Draws polyline connecting predicted points
     - Draws red "X" marker at collision point
     - Draws ghost circle at final predicted position (if no collision)

**Color Coding**:
- Moving balls: Green (from config `highlight_color`)
- Stationary balls: Yellow (0, 255, 255)
- Trajectory: Red (from config `trajectory_color`)
- Collision marker: Red circle with X

---

#### `main()`

**Purpose**: Command-line interface entry point.

**How it works**:
1. Uses `argparse` to parse command-line arguments:
   - `input`: Required video file path
   - `-o, --output`: Optional output video path
   - `-c, --config`: Config file path (default: `../config/config.yaml`)
   - `--headless`: Flag for headless operation

2. Validates input file exists
3. Creates output directory if needed
4. Instantiates `PoolBallTracker` and calls `process_video()`

---

## Ball Detector Module (`ball_detector.py`)

The detector uses computer vision to find pool balls in each frame.

### Class: `BallDetector`

#### `__init__(self, config)`

**Purpose**: Initializes detection parameters from configuration.

**How it works**:
1. **Radius Constraints**:
   - Loads `min_radius` and `max_radius` from config
   - Calculates area bounds: `min_area = π × min_radius² × 0.5` (allows 50% smaller)
   - Calculates `max_area = π × max_radius² × 1.5` (allows 50% larger)
   - These tolerances account for perspective distortion and detection noise

2. **Circularity Constraints**:
   - `min_circ = 0.3`: Allows ovals (perspective distortion)
   - `max_circ = 1.4`: Allows glare/reflection artifacts

3. **Crop Settings**:
   - Extracts crop values (top, bottom, left, right) from config
   - These define the table boundaries (excludes rails, floor, scoreboard)

4. **Debug Mode**: Initialized to `False` (can be enabled for verbose output)

---

#### `detect_balls(self, frame) -> List[Tuple[int, int, int]]`

**Purpose**: Main detection method. Returns list of detected balls as `(x, y, radius)` tuples.

**Processing Pipeline**:

1. **Frame Cropping**:
   ```python
   cropped = frame[top:height-bottom, left:width-right]
   ```
   - Removes unwanted regions (rails, floor, etc.)
   - Only processes the table surface
   - Returns empty list if crop results in zero-size image

2. **Color Space Conversion**:
   - Converts BGR (Blue-Green-Red) to HSV (Hue-Saturation-Value)
   - HSV is better for color-based detection because:
     - Hue represents color type (independent of lighting)
     - Saturation represents color intensity
     - Value represents brightness
   - More robust to lighting variations than RGB

3. **Felt Masking**:
   - Creates mask for felt color using `cv2.inRange()`:
     ```python
     mask_felt = cv2.inRange(hsv, felt_lower, felt_upper)
     ```
   - This creates binary image: white pixels = felt color, black = everything else
   - **Inverts mask**: `mask_objects = bitwise_not(mask_felt)`
   - Now white = non-felt objects (balls, chalk, etc.), black = felt

4. **Morphological Operations** (Noise Reduction):
   - **Opening** (`MORPH_OPEN`): Removes small white specks (salt noise)
     - Erosion followed by dilation
     - Removes tiny false positives
   - **Closing** (`MORPH_CLOSE`): Fills small holes in objects
     - Dilation followed by erosion
     - Connects broken edges (e.g., from ball patterns)
   - Uses 3×3 kernel, 1 iteration opening, 2 iterations closing

5. **Contour Detection**:
   - `cv2.findContours()` finds outlines of white regions
   - `RETR_EXTERNAL`: Only external contours (not nested)
   - `CHAIN_APPROX_SIMPLE`: Compresses contour points

6. **Contour Filtering** (Multi-Stage Validation):
   
   For each contour:
   
   a. **Area Check**:
      - Calculates `cv2.contourArea(contour)`
      - Rejects if outside `[min_area, max_area]` range
      - Filters out noise (too small) and large objects (too big)
   
   b. **Radius Check**:
      - Uses `cv2.minEnclosingCircle()` to find smallest circle containing contour
      - Returns center `(x, y)` and `radius`
      - Rejects if radius outside `[min_radius, max_radius]`
   
   c. **Circularity Check**:
      - Calculates: `circularity = area / (π × radius²)`
      - Perfect circle = 1.0
      - Rejects if outside `[0.3, 1.4]` range
      - Filters out non-circular objects (chalk cubes, shadows)
   
   d. **Coordinate Adjustment**:
      - Adds crop offsets back to get position in full frame:
        ```python
        center_x = int(x) + crop_left
        center_y = int(y) + crop_top
        ```
   
   e. **Brightness Verification**:
      - Calls `_verify_ball()` to check if region is bright enough
      - Prevents detecting shadows or dark pockets as balls

7. **Returns**: List of `(center_x, center_y, radius)` tuples

**Debug Mode**: When enabled, prints detailed filtering information for each contour.

---

#### `_verify_ball(self, frame, x, y, r) -> bool`

**Purpose**: Secondary validation to ensure detected region is actually a ball (not shadow/pocket).

**How it works**:
1. **Edge Check**: Rejects if ball is too close to frame edges (< 5 pixels)
   - Prevents partial detections at boundaries

2. **Mask Creation**:
   - Creates binary mask with circle at `(x, y)` with radius `r-1`
   - Slightly smaller radius to avoid edge pixels

3. **Brightness Analysis**:
   - Converts frame to HSV
   - Calculates mean HSV values within the mask using `cv2.mean()`
   - Extracts Value (brightness) component: `brightness = mean_hsv[2]`

4. **Threshold Check**:
   - Rejects if `brightness < 40` (too dark)
   - Shadows and pockets are typically darker than balls

**Returns**: `True` if passes all checks, `False` otherwise.

---

#### `is_white_ball(self, frame, x, y, r) -> bool`

**Purpose**: Determines if a detected ball is the white cue ball.

**How it works**:
1. Creates circular mask around ball position
2. Calculates mean HSV values within mask
3. Checks if saturation and value fall within white ball range:
   - Low saturation (white has no color)
   - High value (white is bright)
4. Uses config thresholds: `white_ball['lower']` and `white_ball['upper']`

**Note**: Currently defined but not actively used in main pipeline (could be extended for cue ball tracking).

---

## Ball Tracker Module (`ball_tracker.py`)

The tracker maintains persistent identities for balls across frames using distance-based matching.

### Class: `BallTracker`

#### `__init__(self, config)`

**Purpose**: Initializes tracking data structures and parameters.

**How it works**:
1. **ID Management**:
   - `next_object_id = 0`: Counter for assigning unique IDs

2. **Data Structures**:
   - `objects`: `OrderedDict` mapping `object_id → (centroid, radius)`
   - `disappeared`: `OrderedDict` mapping `object_id → frames_missing_count`
   - `position_history`: `OrderedDict` mapping `object_id → deque` of past positions
   - `fast_moving`: `OrderedDict` mapping `object_id → bool` (speed flag)

3. **Configuration Parameters**:
   - `max_disappeared`: Frames before removing lost object
   - `max_distance`: Maximum pixel distance for matching
   - `min_velocity`: Threshold for "moving" classification
   - `pocket_radius` and `pocket_centers`: For pocket detection

4. **History Settings**:
   - `max_history = 10`: Stores last 10 positions per ball (for velocity calculation)

---

#### `register(self, centroid, radius)`

**Purpose**: Adds a new ball to the tracking system.

**How it works**:
1. Assigns current `next_object_id` to the ball
2. Stores `(centroid, radius)` in `objects` dictionary
3. Initializes `disappeared` count to 0
4. Creates empty `deque` in `position_history`
5. Sets `fast_moving` flag to `False`
6. Increments `next_object_id` for next registration

---

#### `deregister(self, object_id)`

**Purpose**: Removes a ball from tracking (e.g., pocketed or lost).

**How it works**:
1. Deletes entry from `objects` dictionary
2. Deletes entry from `disappeared` dictionary
3. Deletes entry from `position_history` (if exists)
4. Deletes entry from `fast_moving` (if exists)

**Note**: Uses `del` statements which raise `KeyError` if ID doesn't exist (shouldn't happen in normal operation).

---

#### `update(self, detections: List[Tuple[int, int, int]], frame_number: int)`

**Purpose**: Core tracking logic. Matches new detections to existing tracked objects.

**How it works**:

**Case 1: No Detections** (Empty frame):
- Increments `disappeared` count for all existing objects
- For fast-moving balls: uses `max_disappeared * 2` patience (more forgiving)
- Deregisters objects that exceed patience threshold
- Returns current tracked objects

**Case 2: First Frame** (No existing objects):
- Registers all detections as new objects
- Returns tracked objects

**Case 3: Matching** (Existing objects + new detections):

1. **Data Preparation**:
   - Converts detections to NumPy arrays:
     - `input_centroids`: Array of `(x, y)` coordinates
     - `input_radii`: Array of radius values
   - Extracts existing object centroids

2. **Distance Matrix Calculation**:
   - Uses `scipy.spatial.distance.cdist()` to compute pairwise distances
   - Creates matrix `D` where `D[i, j]` = distance between object `i` and detection `j`

3. **Greedy Matching Algorithm**:
   - Sorts rows by minimum distance (closest matches first)
   - For each row (existing object):
     - Finds column (detection) with minimum distance
     - Checks if distance < threshold:
       - Fast-moving balls: `threshold = max_distance * 1.5` (more forgiving)
       - Normal balls: `threshold = max_distance`
     - If valid match:
       - Updates object position and radius
       - Resets `disappeared` count to 0
       - Marks row and column as "used"
     - Skips if already matched

4. **Unmatched Objects**:
   - Objects not matched → increment `disappeared` count
   - Deregister if exceeds patience (with fast-moving adjustment)

5. **Unmatched Detections**:
   - New detections not matched → register as new objects

6. **Pocket Detection**:
   - Checks if any ball's centroid is inside a pocket (using `_is_in_pocket()`)
   - Deregisters pocketed balls immediately

7. **History Update**:
   - Appends current position to `position_history` for all active objects
   - Stores as `(x, y, frame_number)` tuple

8. **Returns**: Calls `get_tracked_objects()` to return enriched data

**Algorithm Choice**: Greedy matching (closest-first) is simple and fast. More sophisticated algorithms (Hungarian algorithm) could handle complex cases better but add complexity.

---

#### `_is_in_pocket(self, x: float, y: float) -> bool`

**Purpose**: Checks if a point is inside any pocket circle.

**How it works**:
1. Iterates through all pocket centers from config
2. Calculates Euclidean distance: `√((x - px)² + (y - py)²)`
3. Returns `True` if distance < `pocket_radius` for any pocket

**Note**: Uses ball center, not edge. Ball is considered pocketed when center enters pocket.

---

#### `get_tracked_objects(self, frame_number: int) -> Dict`

**Purpose**: Returns enriched tracking data with velocities and movement status.

**How it works**:
1. Iterates through all active objects
2. For each object:
   - Calls `calculate_velocity()` to get `(vx, vy)` vector
   - Calculates speed: `speed = √(vx² + vy²)`
   - Classifies as "fast" if `speed > min_velocity * 5`
   - Updates `fast_moving` flag
   - Determines `is_moving` if `speed > min_velocity`

3. Returns dictionary:
   ```python
   {
       object_id: {
           'centroid': (x, y),
           'radius': r,
           'velocity': (vx, vy),
           'speed': float,
           'is_moving': bool,
           'is_fast': bool
       },
       ...
   }
   ```

---

#### `calculate_velocity(self, object_id: int, current_frame: int) -> Tuple[float, float]`

**Purpose**: Calculates velocity vector using weighted average of recent positions.

**How it works**:

**Method 1: Weighted Average** (if history ≥ 3 positions):
1. Iterates backwards through last 5 positions (or all if fewer)
2. For each pair of consecutive positions:
   - Calculates velocity: `vx = (x2 - x1) / (frame2 - frame1)`
   - Calculates weight: `weight = 1.0 / (total_positions - index)`
   - More recent positions have higher weight
3. Computes weighted average:
   ```python
   vx = Σ(vx_i × weight_i) / Σ(weight_i)
   ```

**Method 2: Simple Difference** (if history < 3):
- Uses only last two positions
- `vx = (x2 - x1) / (frame2 - frame1)`
- Returns `(0, 0)` if frames are equal (division by zero protection)

**Why Weighted**: Recent positions are more indicative of current velocity. Older positions may reflect past motion that's no longer relevant.

**Returns**: `(vx, vy)` tuple in pixels per frame.

---

## Trajectory Predictor Module (`trajectory_predictor.py`)

The predictor simulates ball physics to forecast future positions.

### Class: `TrajectoryPredictor`

#### `__init__(self, config, frame_width: int, frame_height: int)`

**Purpose**: Initializes physics parameters and table boundaries.

**How it works**:
1. **Physics Parameters** (from config):
   - `friction_coefficient`: Multiplier applied each frame (e.g., 0.98 = 2% speed loss)
   - `prediction_frames`: How many frames into future to simulate
   - `cushion_bounce`: Energy retention after wall bounce (e.g., 0.7 = 70% speed retained)

2. **Table Boundaries**:
   - Calculates cushion positions from crop settings:
     ```python
     left_cushion = crop_left
     right_cushion = width - crop_right
     top_cushion = crop_top
     bottom_cushion = height - crop_bottom
     ```
   - These define the playable area

3. **Pocket Configuration**:
   - Loads pocket radius and center coordinates from config
   - Used to detect when ball enters pocket (stops prediction)

---

#### `predict_trajectory(self, position, velocity, radius) -> List[Tuple[int, int]]`

**Purpose**: Simulates ball motion forward in time, returning predicted path points.

**How it works**:

1. **Initialization**:
   - Starts at current `position` `(x, y)`
   - Starts with current `velocity` `(vx, vy)`

2. **Simulation Loop** (for `prediction_frames` iterations):
   
   a. **Apply Friction**:
      ```python
      vx *= friction_coefficient
      vy *= friction_coefficient
      ```
      - Reduces velocity each frame (simulates table friction)
      - Exponential decay: `v(t) = v₀ × friction^t`
   
   b. **Speed Check**:
      - Calculates `speed = √(vx² + vy²)`
      - If `speed < 0.1`: breaks loop (ball essentially stopped)
   
   c. **Move Ball**:
      ```python
      x += vx
      y += vy
      ```
      - Updates position based on velocity
   
   d. **Pocket Check**:
      - Calls `_is_in_pocket(x, y)` to check if center entered pocket
      - If pocketed: appends final point and breaks (ball disappears)
   
   e. **Wall Collision Detection** (4 walls):
      
      **Left Wall** (`x - radius <= left_cushion`):
      - Resets position: `x = left_cushion + radius` (prevents penetration)
      - Reverses X velocity: `vx = -vx × cushion_bounce`
      
      **Right Wall** (`x + radius >= right_cushion`):
      - Resets position: `x = right_cushion - radius`
      - Reverses X velocity: `vx = -vx × cushion_bounce`
      
      **Top Wall** (`y - radius <= top_cushion`):
      - Resets position: `y = top_cushion + radius`
      - Reverses Y velocity: `vy = -vy × cushion_bounce`
      
      **Bottom Wall** (`y + radius >= bottom_cushion`):
      - Resets position: `y = bottom_cushion - radius`
      - Reverses Y velocity: `vy = -vy × cushion_bounce`
      
      **Note**: Checks X and Y independently (allows corner bounces)
   
   f. **Record Point**:
      - Appends `(int(x), int(y))` to trajectory list

3. **Returns**: List of `(x, y)` tuples representing predicted path

**Physics Model**: Simplified 2D physics with:
- Constant friction (no air resistance)
- Perfect elastic collisions with energy loss
- No angular momentum (no spin effects)
- Instantaneous bounces (no deformation)

---

#### `_is_in_pocket(self, x: float, y: float) -> bool`

**Purpose**: Checks if a point is inside any pocket (same as tracker's method).

**How it works**:
- Identical to `BallTracker._is_in_pocket()`
- Calculates distance to each pocket center
- Returns `True` if within `pocket_radius` of any pocket

---

#### `find_collision_point(self, trajectory: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]`

**Purpose**: Finds the first point in trajectory where ball hits a cushion.

**How it works**:
1. Iterates through trajectory points
2. For each point `(x, y)`, checks proximity to any wall:
   - `abs(x - left_cushion) < 10` OR
   - `abs(x - right_cushion) < 10` OR
   - `abs(y - top_cushion) < 10` OR
   - `abs(y - bottom_cushion) < 10`
3. Returns first point that meets criteria
4. If no collision found, returns last point in trajectory

**Note**: 10-pixel threshold accounts for discrete simulation steps.

---

#### `estimate_time_to_collision(self, trajectory: List[Tuple[int, int]]) -> Optional[float]`

**Purpose**: Calculates how many frames until collision occurs.

**How it works**:
1. Calls `find_collision_point()` to get collision location
2. Finds index of that point in trajectory list
3. Since trajectory is ordered by time (frame 0, 1, 2...), index = frame number
4. Returns index as float (time in frames)

**Returns**: `None` if no collision found or trajectory empty.

---

#### `predict_ball_collision(self, ball1_pos, ball1_vel, ball1_radius, ball2_pos, ball2_vel, ball2_radius) -> Optional[Tuple[int, int, float]]`

**Purpose**: Predicts if and when two balls will collide.

**How it works**:

1. **Setup**:
   - Unpacks positions: `(x1, y1)`, `(x2, y2)`
   - Unpacks velocities: `(vx1, vy1)`, `(vx2, vy2)`
   - Calculates minimum distance for collision: `min_dist = radius1 + radius2`

2. **Forward Simulation**:
   - Loops through future frames `t = 0` to `prediction_frames`
   - For each time `t`:
     
     a. **Predict Ball 1 Position**:
        ```python
        pred_x1 = x1 + vx1 * t * (friction ** t)
        pred_y1 = y1 + vy1 * t * (friction ** t)
        ```
        - Accounts for friction: velocity decreases over time
        - `friction ** t` = exponential decay
     
     b. **Predict Ball 2 Position**:
        ```python
        pred_x2 = x2 + vx2 * t * (friction ** t)
        pred_y2 = y2 + vy2 * t * (friction ** t)
        ```
     
     c. **Calculate Distance**:
        ```python
        dist = √((pred_x1 - pred_x2)² + (pred_y1 - pred_y2)²)
        ```
     
     d. **Collision Check**:
        - If `dist <= min_dist`: balls are touching
        - Calculates collision point: midpoint between centers
        - Returns `(collision_x, collision_y, time_t)`

3. **Returns**: `None` if no collision within prediction window

**Limitations**:
- Assumes both balls maintain constant velocity direction (ignores wall bounces)
- Simplified friction model (may not match actual trajectory)
- No collision response calculation (just detection)

**Use Case**: Used in `main.py` to predict ball-to-ball collisions and visualize with red "X" markers.

---

## Test Detection Module (`test_detection.py`)

Utility script for testing and debugging detection on single frames.

### Function: `visualize_detections(config_path, video_path, output_path, frame_num, crop_test_only)`

**Purpose**: Loads a single frame and either tests crop settings or runs detection visualization.

**How it works**:

1. **Configuration Loading**:
   - Validates config file exists
   - Loads YAML config

2. **Video Frame Extraction**:
   - Opens video with `cv2.VideoCapture()`
   - Seeks to specified frame number by reading frames sequentially
   - Extracts single frame
   - Releases video capture

3. **Crop Test Mode** (`crop_test_only=True`):
   - Reads crop values from config
   - Calculates crop boundaries:
     ```python
     y1 = top
     y2 = height - bottom
     x1 = left
     x2 = width - right
     ```
   - Draws thick white rectangle on frame showing crop area
   - Saves to `../output/test_crop.jpg`
   - **Purpose**: Helps user verify crop settings visually

4. **Detection Mode** (`crop_test_only=False`):
   - Creates `BallDetector` instance
   - Enables debug mode
   - Calls `detect_balls()` on frame
   - Visualizes results:
     - Green circle (20px radius) around each detection
     - Red dot at center
     - Text label with coordinates and detected radius
   - Saves to output path
   - Prints detection summary to console

**Usage**: Run from command line or modify `CROP_TEST` flag in script.

---

## Configuration System

The system uses YAML configuration for all tunable parameters.

### File: `config/config.yaml`

**Structure**:

1. **`ball_detection`**:
   - `min_radius` / `max_radius`: Size constraints for ball detection
   - `felt_color`: HSV range for table felt (used for masking)
   - `white_ball`: HSV range for cue ball identification
   - `colored_balls`: HSV range for numbered balls

2. **`tracking`**:
   - `max_disappeared`: Frames before removing lost object
   - `max_distance`: Maximum pixel distance for matching detections
   - `min_velocity`: Speed threshold for "moving" classification

3. **`trajectory`**:
   - `friction_coefficient`: Speed decay per frame (0.98 = 2% loss)
   - `prediction_frames`: How many frames ahead to simulate
   - `cushion_bounce`: Energy retention after wall bounce (0.7 = 70%)

4. **`video`**:
   - `output_fps`: FPS for output video
   - `show_velocity_text`: Display velocity in HUD
   - `show_trajectory_line`: Draw predicted path lines
   - `highlight_color`: RGB color for moving balls
   - `trajectory_color`: RGB color for trajectory lines
   - `crop`: Pixel values to crop from edges

5. **`pockets`**:
   - `radius`: Pocket detection radius in pixels
   - `locations`: List of `[x, y]` coordinates for each pocket

**Why YAML**: Human-readable, supports nested structures, easy to edit without code changes.

---

## Data Flow Summary

```
Video Frame (BGR image)
    ↓
[BallDetector.detect_balls()]
    ├─ Crop frame
    ├─ Convert BGR → HSV
    ├─ Create felt mask → invert
    ├─ Morphological operations
    ├─ Find contours
    └─ Filter by area/radius/circularity
    ↓
List of (x, y, radius) detections
    ↓
[BallTracker.update()]
    ├─ Calculate distance matrix
    ├─ Greedy matching algorithm
    ├─ Update object positions
    ├─ Check for pocketed balls
    ├─ Update position history
    └─ Calculate velocities
    ↓
Dictionary of tracked objects with velocities
    ↓
[TrajectoryPredictor.predict_trajectory()] (for each moving ball)
    ├─ Simulate physics forward
    ├─ Apply friction
    ├─ Check wall collisions
    └─ Check pocket entry
    ↓
Predicted trajectory points
    ↓
[PoolBallTracker.draw_tracking()]
    ├─ Draw HUD overlay
    ├─ Draw ball circles
    ├─ Draw trajectory lines
    ├─ Draw collision markers
    └─ Add text labels
    ↓
Annotated frame (BGR image with overlays)
    ↓
Output video / Display
```

---

## Key Design Patterns

1. **Separation of Concerns**: Each module has a single responsibility
   - Detection: Finding balls
   - Tracking: Maintaining identities
   - Prediction: Physics simulation
   - Main: Orchestration and visualization

2. **Configuration-Driven**: All parameters in YAML (no hardcoding)

3. **Frame-by-Frame Processing**: Simple, predictable pipeline

4. **Greedy Matching**: Fast, good enough for most cases

5. **History-Based Velocity**: Weighted average for smooth, accurate speed calculation

6. **Simplified Physics**: Fast simulation with acceptable accuracy

---

## Limitations and Future Improvements

**Current Limitations**:
- No ball-to-ball collision response (only detection)
- No spin/angular momentum
- Simplified friction model
- Greedy matching may fail with many balls
- No occlusion handling (balls behind each other)

**Potential Enhancements**:
- Hungarian algorithm for optimal matching
- Kalman filtering for smoother tracking
- 3D physics simulation
- Machine learning for detection
- Real-time processing optimization

---

## Conclusion

The Pool Ball Tracker demonstrates a complete computer vision pipeline combining:
- **Image Processing**: Color space conversion, masking, morphology
- **Object Detection**: Contour analysis and filtering
- **Multi-Object Tracking**: Distance-based matching with persistence
- **Physics Simulation**: Friction, collisions, and trajectory prediction
- **Visualization**: Real-time HUD and overlay rendering

Each component is designed to be modular, configurable, and understandable, making the system both functional and maintainable.

