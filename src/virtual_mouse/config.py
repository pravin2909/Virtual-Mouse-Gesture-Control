"""Central configuration for the virtual mouse.

All tunable parameters live here so behaviour can be adjusted in one place.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Runtime configuration values."""

    # Camera capture resolution.
    camera_width: int = 1280
    camera_height: int = 720
    camera_index: int = 0

    # MediaPipe hand-tracking settings.
    model_complexity: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    max_num_hands: int = 1

    # Cursor smoothing: number of samples in the moving-average window.
    smoothing_window: int = 5

    # Gesture detection thresholds.
    fingers_together_threshold: float = 0.05
    gesture_cooldown_seconds: float = 0.3

    # Main loop pacing.
    loop_delay_seconds: float = 0.01

    # Key that quits the application.
    quit_key: str = "q"


# Default configuration instance used across the app.
DEFAULT_CONFIG = Config()
