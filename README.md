# Pool Ball Tracker 🎱

A computer vision system that tracks pool balls in video, calculates their velocity, and predicts their trajectories including cushion collisions.

## Features

- **Ball Detection**: Automatically detects pool balls using color segmentation and Hough Circle Transform
- **Multi-Ball Tracking**: Tracks multiple balls simultaneously across frames
- **Velocity Calculation**: Computes velocity (speed and direction) for each moving ball
- **Trajectory Prediction**: Projects where balls will hit cushions, accounting for friction and bouncing
- **Visual Feedback**: Highlights moving balls, shows velocity vectors, and draws predicted trajectories

## Quick Start

### Using GitHub Codespaces (Easiest!)

1. Push this code to GitHub
2. Open your repo on GitHub
3. Click "Code" → "Codespaces" → "Create codespace on main"
4. Once it loads, run:
```bash# pool-ball-tracker
