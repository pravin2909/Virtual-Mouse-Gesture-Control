"""Application entry point: captures video and runs the gesture loop."""

import time

import cv2
import mediapipe as mp

from .config import DEFAULT_CONFIG
from .gestures import GestureDetector
from .mouse_controller import MouseController


class VirtualMouseApp:
    """Wires together video capture, hand tracking and gesture control."""

    def __init__(self, config=DEFAULT_CONFIG):
        self._config = config
        self._hands_solution = mp.solutions.hands
        self._hands = self._hands_solution.Hands(
            static_image_mode=False,
            model_complexity=config.model_complexity,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            max_num_hands=config.max_num_hands,
        )
        self._draw = mp.solutions.drawing_utils
        self._detector = GestureDetector(
            MouseController(smoothing_window=config.smoothing_window),
            config,
        )

    def run(self):
        """Start the capture loop. Press the configured quit key to stop."""
        cap = cv2.VideoCapture(self._config.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.camera_height)
        quit_key = ord(self._config.quit_key)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = self._hands.process(frame_rgb)

                landmarks = []
                if processed.multi_hand_landmarks:
                    hand_landmarks = processed.multi_hand_landmarks[0]
                    self._draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self._hands_solution.HAND_CONNECTIONS,
                    )
                    landmarks = list(hand_landmarks.landmark)

                self._detector.process(frame, landmarks, processed)

                time.sleep(self._config.loop_delay_seconds)
                cv2.imshow("Virtual Mouse", frame)
                if cv2.waitKey(1) & 0xFF == quit_key:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    """Console-script entry point."""
    VirtualMouseApp().run()


if __name__ == "__main__":
    main()
