# Pool Ball Tracker 🎱

Welcome to the Pool Ball Tracker!

This is a computer vision project designed to help you "see" the physics of pool. It watches video of a pool table, finds the balls, tracks their speed, and even predicts where they are going to bounce!

Whether you are a coder or a pool player, this tool helps you visualize the game in a whole new way.

## What Does It Do?

* **👀 Finds the Balls:** You don't need to tell it where the balls are. It uses smart image processing to spot them on the felt automatically.
* **🕵️ Keeps Track:** It remembers which ball is which as they slide across the table, even when multiple balls are moving at once.
* **⚡ Shows Speed:** It calculates exactly how fast each ball is moving and displays it on the screen.
* **🔮 Predicts the Future:** It draws a line showing exactly where the ball is headed and how it will bounce off the cushions (rails).
* **🎨 Visual HUD:** It overlays all this data right on the video, looking like a pro sports broadcast or a video game.

## Quick Start Guide

The easiest way to run this is using **GitHub Codespaces**. This sets up a computer in the cloud for you, so you don't have to worry about installing complex software on your own machine.

### Step 1: Open in Codespaces
1.  Push this code to your GitHub account (if you haven't already).
2.  Click the green **Code** button near the top right of the repository page.
3.  Select the **Codespaces** tab.
4.  Click the big green button that says **Create codespace on main**.

### Step 2: Run the Tracker
Once the screen loads and you see a terminal (a text box) at the bottom:

1.  **Install the tools:** Copy and paste this command into the terminal and hit Enter:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the program:** Copy and paste this command to start tracking:
    ```bash
    python src/main.py data/videos/pool_fixed.mp4
    ```

## Want to use your own video? 📹

You can absolutely use your own clips! Just follow these three rules to make it work:

### 1. It MUST be "Overhead"
The camera needs to be looking straight down at the table (a bird's-eye view).
* ✅ **Good:** A drone shot or a camera mounted on the ceiling.
* ❌ **Bad:** A video taken from the side while standing next to the table. (The math won't work if the table looks like a trapezoid!)

### 2. Update the "Crop" Settings
The code needs to know exactly where the table is so it doesn't accidentally track shoes or chairs on the floor.
1.  Open the file `config/config.yaml`.
2.  Look for the `crop` section.
3.  Adjust the `top`, `bottom`, `left`, and `right` numbers. These are the number of pixels to "cut off" from each side of the video until ONLY the blue/green felt is visible.

### 3. Run with your file path
Instead of the default command, tell Python where your new video is:
```bash
python src/main.py path/to/your/new_video.mp4
