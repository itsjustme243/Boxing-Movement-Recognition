# Boxing Movement Recognition: My First Steps into Computer Vision

Hi! I'm Jakub. I'm a first-year Data Engineering student and I also happen to love sports.

This project started because I wanted to explore how we can use cameras and data to track human movement. My ultimate goal is to build an application that measures the speed of a punch in a 3D space. 

Since that is a massive and complicated topic, I decided to start small by creating this initial project: detecting a left jab.

This repository contains two different approaches I tried, and honestly, it was a huge reality check for me regarding how AI actually works.

## Approach 1: My own math logic (The one that actually works)
Files: `math.ipynb`

Before messing around with neural networks, I wanted to see if I could solve this using just geometry. 
* What I did: I used MediaPipe to get the x, y, and z coordinates of my shoulder, elbow, and wrist. Then I used basic trigonometry (the Law of Cosines) to calculate the arm extension angle in real-time and tracked the velocity of the wrist.
* How it turned out: From my observations, it works pretty well! The proportion between slow and fast jabs is quite accurate. The jab counter is reliable and detects about 9/10 punches (from a 2D perspective, when I am standing sideways to the camera).

## Approach 2: Trying to force AI into it (Experimental)
Files: `collect_data.py`, `train.py`, `main.py`

(Note: I used a lot of AI tools to help me build this part, as I don't fully understand how to code neural networks from scratch yet).

Since I want to build the most efficient and accurate system, I thought about using neural networks: 
* How it works: I recorded 30 videos of myself throwing jabs and 30 videos of just standing in guard to create a small dataset. Then I trained a 3-layer LSTM model in TensorFlow to recognize the sequence of movements.
* Results: To be completely honest, it works pretty badly compared to the math version. It gets easily confused by different camera angles. Based on my testing, the model only understood one specific movement—when I'm standing with my right side to the camera and punching straight—and even then, with very low accuracy.

## What this project taught me

As a first-year student, this was a huge lesson in computer vision. I realized two things:
1. AI isn't magic. A neural network (like my LSTM) is only as good as the data you give it. My 60 video clips were nowhere near enough to train a stable AI. 
2. Simple math is often better than complex AI. For this specific task, my custom geometric shortcuts beat a heavy deep learning model on every level.

I am keeping both versions here because I am proud of the progress and the mistakes I made. It taught me how to actually think like a data engineer, not just someone who blindly copies code.

## Prerequisites & Installation

If you want to try it out, you'll need a webcam and these Python libraries:

```bash
pip install opencv-python mediapipe numpy tensorflow scikit-learn