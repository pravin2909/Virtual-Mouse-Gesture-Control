# Virtual Mouse Gesture Control

## Overview

The **Virtual Mouse Gesture Control** project allows you to control your computer's mouse using hand gestures, using a webcam for hand tracking. This project uses **MediaPipe** for hand tracking, **OpenCV** for image processing, and **pyautogui** for mouse control. It enables you to move the mouse and perform actions such as single-click and double-click by detecting specific gestures made with your fingers.

## Features

- **Mouse Movement**: Move the mouse cursor on the screen using your index finger.
- **Single Click**: Perform a left-click when your thumb and index finger come together.
- **Double Click**: Perform a double-click when your index and middle fingers come together.

The system uses hand landmarks to detect the positioning of the thumb, index, and middle fingers and performs mouse actions based on their proximity.

## Tools and Libraries Used

- **MediaPipe**: A framework for building pipelines to process video and track hand landmarks.
- **OpenCV**: A computer vision library for capturing the webcam feed and performing image processing.
- **PyAutoGUI**: A module for controlling the mouse and performing actions like clicking and moving the pointer.
- **Pynput**: A library for simulating mouse events.
- **NumPy**: A fundamental package for scientific computing in Python, used for calculating distances and angles.
