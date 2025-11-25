Pool Ball Tracker 🎱
Welcome to the Pool Ball Tracker!

This is a computer vision project designed to help you "see" the physics of pool. It watches video of a pool table, finds the balls, tracks their speed, and even predicts where they are going to bounce!

Whether you are a coder or a pool player, this tool helps you visualize the game in a whole new way.

What Does It Do?
👀 Finds the Balls: You don't need to tell it where the balls are. It uses smart image processing to spot them on the felt automatically.

🕵️ Keeps Track: It remembers which ball is which as they slide across the table, even when multiple balls are moving at once.

⚡ Shows Speed: It calculates exactly how fast each ball is moving and displays it on the screen.

🔮 Predicts the Future: It draws a line showing exactly where the ball is headed and how it will bounce off the cushions (rails).

🎨 Visual HUD: It overlays all this data right on the video, looking like a pro sports broadcast or a video game.

Quick Start Guide
The easiest way to run this is using GitHub Codespaces. This sets up a computer in the cloud for you, so you don't have to worry about installing complex software on your own machine.

Step 1: Open in Codespaces
Push this code to your GitHub account (if you haven't already).

Click the green Code button near the top right of the repository page.

Select the Codespaces tab.

Click the big green button that says Create codespace on main.

Step 2: Run the Tracker
Once the screen loads and you see a terminal (a text box) at the bottom:

Install the tools: Copy and paste this command into the terminal and hit Enter:

Bash

pip install -r requirements.txt
Run the program: Copy and paste this command to start tracking:

Bash

# Note: Make sure you run this from the 'src' folder!
cd src
python main.py ../data/videos/pool_fixed.mp4 -o ../output/tracked_game.mp4 --headless
View the Result:

Open the file explorer on the left.

Navigate to the output folder.

Right-click tracked_game.mp4 and select Download.

⚙️ Configuration: Using Your Own Videos
You can absolutely use your own clips! Just follow these steps to ensure the tracker understands your video.

1. Video Requirements
The camera must be Overhead. It needs to be looking straight down at the table (a bird's-eye view).

✅ Good: A drone shot or a camera mounted on the ceiling.

❌ Bad: A video taken from the side. (The math won't work if the table looks like a trapezoid!)

2. Setting up the "Crop" (Critical Step)
The tracking code needs to know exactly where the table surface is. We need to "crop out" the floor, the scoreboards, and the rails so the computer only sees the blue/green felt.

How to find the perfect crop:

Open src/test_detection.py.

Scroll to the bottom and find the Manual Switch:

Python

# SET THIS TO TRUE to find your crop values
CROP_TEST = True
Run the test script in your terminal:

Bash

cd src
python test_detection.py
Check the output: Open output/test_crop.jpg. You will see your video frame with a thick white box drawn on it.

Adjust Config:

If the box includes the floor or scoreboard -> Increase the numbers in config/config.yaml.

If the box cuts off part of the table -> Decrease the numbers.

Example config:

YAML

video:
  crop:
    top: 80      # Pixels to remove from top
    bottom: 250  # Pixels to remove from bottom (removes scoreboard)
    left: 100    # Pixels to remove from left
    right: 100   # Pixels to remove from right
Repeat steps 3-5 until the white box perfectly surrounds only the felt.

3. Finding the Pockets
Once your crop is perfect, you need to tell the physics engine where the pockets are (so the prediction line stops when a ball goes in!).

Open your output/test_crop.jpg in an image editor (like Paint or Photoshop) or an online tool like Photopea.

Hover your mouse over the center of each pocket.

Write down the (x, y) coordinates.

Update the pockets section in config/config.yaml:

YAML

pockets:
  radius: 45
  locations:
    - [350, 220]  # Replace these with YOUR coordinates
    - [960, 210]
    # ... add all 6 pockets
4. Run the Real Tracker!
Once your crop is set, go back to src/test_detection.py and turn off the test mode:

Python

# SET THIS TO FALSE to run detection
CROP_TEST = False
Now you are ready to run main.py and see the magic happen!
