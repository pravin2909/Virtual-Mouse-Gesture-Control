# Virtual Mouse Gesture Control

Control your computer's mouse using hand gestures captured through a webcam.
The project uses **MediaPipe** for hand tracking, **OpenCV** for image
processing, and **PyAutoGUI** / **Pynput** for mouse control.

## Features

- **Mouse Movement** — move the cursor by pointing with your index finger.
- **Single Click** — bring your thumb and index finger together.
- **Double Click** — bring your index and middle fingers together.

Hand landmarks are used to detect the position of the thumb, index, and middle
fingers, and mouse actions are triggered based on their proximity. Cursor
motion is smoothed with a moving-average filter for steadier tracking.

## Project Structure

```
Virtual-Mouse-Gesture-Control/
├── main.py                       # Convenience launcher (python main.py)
├── pyproject.toml                # Packaging metadata & dependencies
├── requirements.txt              # Pinned dependencies
├── README.md
└── src/
    └── virtual_mouse/
        ├── __init__.py
        ├── config.py             # Tunable settings (camera, thresholds, …)
        ├── filters.py            # MovingAverageFilter for cursor smoothing
        ├── geometry.py           # Landmark distance / finger-state helpers
        ├── mouse_controller.py   # Cursor movement & click actions
        ├── gestures.py           # Gesture recognition logic
        └── app.py                # Capture loop & entry point
```

## Installation

Requires **Python 3.8+** and a webcam.

```bash
# Clone the repository
git clone <repo-url>
cd Virtual-Mouse-Gesture-Control

# (Recommended) create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Alternatively, install as a package (this also registers a `virtual-mouse`
command):

```bash
pip install -e .
```

## Usage

Run with the launcher script:

```bash
python main.py
```

Or, if installed as a package:

```bash
virtual-mouse
```

Press **`q`** in the video window to quit.

## Configuration

All tunable parameters — camera resolution, detection confidence, smoothing
window, gesture thresholds and cooldown — live in
[`src/virtual_mouse/config.py`](src/virtual_mouse/config.py). Adjust the values
in the `Config` dataclass to change behaviour.

## Tools and Libraries

- **MediaPipe** — hand-landmark tracking pipeline.
- **OpenCV** — webcam capture and image processing.
- **PyAutoGUI** — cursor movement.
- **Pynput** — mouse click simulation.
- **NumPy** — distance calculations and smoothing.

## License

Released under the MIT License.
