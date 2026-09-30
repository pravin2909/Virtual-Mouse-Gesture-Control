"""Signal-smoothing filters used to steady cursor movement."""

import numpy as np


class MovingAverageFilter:
    """Smooths a stream of values using a fixed-size moving average window."""

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = []

    def update(self, value):
        """Add a new value and return the current windowed average."""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return np.mean(self.values, axis=0)
