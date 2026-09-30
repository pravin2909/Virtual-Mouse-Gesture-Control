"""Translates hand landmarks into on-screen mouse actions."""

import pyautogui
from pynput.mouse import Button, Controller

from .filters import MovingAverageFilter


class MouseController:
    """Moves the cursor and issues clicks based on smoothed landmark input."""

    def __init__(self, smoothing_window=5):
        self._mouse = Controller()
        self.screen_width, self.screen_height = pyautogui.size()
        self._filter_x = MovingAverageFilter(window_size=smoothing_window)
        self._filter_y = MovingAverageFilter(window_size=smoothing_window)

    def move_to(self, index_finger_tip):
        """Move the cursor to the smoothed position of the index-finger tip."""
        if index_finger_tip is None:
            return
        x = int(index_finger_tip.x * self.screen_width)
        y = int(index_finger_tip.y * self.screen_height)
        x_smooth = int(self._filter_x.update(x))
        y_smooth = int(self._filter_y.update(y))
        pyautogui.moveTo(x_smooth, y_smooth)

    def left_click(self):
        """Perform a single left click."""
        self._mouse.click(Button.left, 1)

    def double_click(self):
        """Perform a double left click."""
        self._mouse.click(Button.left, 2)
