"""Gesture recognition: maps hand landmarks to mouse actions."""

import time

import cv2
import mediapipe as mp

from .geometry import are_fingers_together, is_finger_up

_HAND_LANDMARK = mp.solutions.hands.HandLandmark


class GestureDetector:
    """Detects supported gestures and drives the mouse controller.

    Supported gestures:
        * Index finger up  -> move cursor
        * Thumb + index together -> single left click
        * Index + middle together -> double left click
    """

    def __init__(self, mouse_controller, config):
        self._mouse = mouse_controller
        self._config = config
        self._last_gesture_time = 0.0

    def _cooldown_elapsed(self, current_time):
        return (
            current_time - self._last_gesture_time
            > self._config.gesture_cooldown_seconds
        )

    def process(self, frame, landmarks, processed):
        """Interpret ``landmarks`` for the current ``frame`` and act on them."""
        if len(landmarks) < 21:
            return

        hand = processed.multi_hand_landmarks[0].landmark
        index_finger_tip = hand[_HAND_LANDMARK.INDEX_FINGER_TIP]
        thumb_tip = hand[_HAND_LANDMARK.THUMB_TIP]
        middle_finger_tip = hand[_HAND_LANDMARK.MIDDLE_FINGER_TIP]

        index_finger_up = is_finger_up(
            landmarks,
            _HAND_LANDMARK.INDEX_FINGER_TIP,
            _HAND_LANDMARK.INDEX_FINGER_PIP,
        )
        middle_finger_up = is_finger_up(
            landmarks,
            _HAND_LANDMARK.MIDDLE_FINGER_TIP,
            _HAND_LANDMARK.MIDDLE_FINGER_PIP,
        )

        threshold = self._config.fingers_together_threshold
        thumb_and_index_together = are_fingers_together(
            thumb_tip, index_finger_tip, threshold=threshold
        )
        index_and_middle_together = are_fingers_together(
            index_finger_tip, middle_finger_tip, threshold=threshold
        )

        if index_finger_up:
            self._mouse.move_to(index_finger_tip)

        current_time = time.time()

        if thumb_and_index_together and index_finger_up:
            if self._cooldown_elapsed(current_time):
                self._mouse.left_click()
                self._draw_label(frame, "Single Click", y=100)
                self._last_gesture_time = current_time

        if index_and_middle_together and index_finger_up and middle_finger_up:
            if self._cooldown_elapsed(current_time):
                self._mouse.double_click()
                self._draw_label(frame, "Double Click", y=150)
                self._last_gesture_time = current_time

    @staticmethod
    def _draw_label(frame, text, y):
        cv2.putText(
            frame,
            text,
            (50, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
