"""Geometric helpers for interpreting hand landmarks."""

import numpy as np


def get_distance(a, b):
    """Return the Euclidean distance between two ``(x, y)`` points."""
    x1, y1 = a
    x2, y2 = b
    return np.hypot(x2 - x1, y2 - y1)


def is_finger_up(landmarks, finger_tip_id, finger_base_id):
    """Return ``True`` when a finger tip sits above its base (finger extended).

    Image coordinates run top-to-bottom, so a smaller ``y`` means higher up.
    """
    return landmarks[finger_tip_id].y < landmarks[finger_base_id].y


def are_fingers_together(landmark1, landmark2, threshold=0.05):
    """Return ``True`` when two landmarks are closer than ``threshold``."""
    coords1 = (landmark1.x, landmark1.y)
    coords2 = (landmark2.x, landmark2.y)
    return get_distance(coords1, coords2) < threshold
