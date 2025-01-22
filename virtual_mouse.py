import cv2
import mediapipe as mp
import pyautogui
from pynput.mouse import Button, Controller
import time
import numpy as np

mouse = Controller()
screen_width, screen_height = pyautogui.size()
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = []

    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return np.mean(self.values, axis=0)

filter_x = MovingAverageFilter(window_size=5)
filter_y = MovingAverageFilter(window_size=5)

def get_distance(a, b):
    x1, y1 = a
    x2, y2 = b
    return np.hypot(x2 - x1, y2 - y1)

def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)
        x_smooth = int(filter_x.update(x))
        y_smooth = int(filter_y.update(y))
        pyautogui.moveTo(x_smooth, y_smooth)

def is_finger_up(landmarks, finger_tip_id, finger_base_id):
    return landmarks[finger_tip_id].y < landmarks[finger_base_id].y

def are_fingers_together(landmark1, landmark2, threshold=0.05):
    coords1 = (landmark1.x, landmark1.y)
    coords2 = (landmark2.x, landmark2.y)
    distance = get_distance(coords1, coords2)
    return distance < threshold

def detect_gesture(frame, landmarks, processed):
    global last_click_time, click_count, last_gesture_time

    if len(landmarks) >= 21:
        index_finger_tip = processed.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = processed.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.THUMB_TIP]
        middle_finger_tip = processed.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP]

        index_finger_up = is_finger_up(landmarks, mpHands.HandLandmark.INDEX_FINGER_TIP, mpHands.HandLandmark.INDEX_FINGER_PIP)
        middle_finger_up = is_finger_up(landmarks, mpHands.HandLandmark.MIDDLE_FINGER_TIP, mpHands.HandLandmark.MIDDLE_FINGER_PIP)

        thumb_and_index_together = are_fingers_together(thumb_tip, index_finger_tip, threshold=0.05)
        index_and_middle_together = are_fingers_together(index_finger_tip, middle_finger_tip, threshold=0.05)

        if index_finger_up:
            move_mouse(index_finger_tip)

        current_time = time.time()

        if thumb_and_index_together and index_finger_up:
            if current_time - last_gesture_time > 0.3:
                mouse.click(Button.left, 1)
                cv2.putText(frame, "Single Click", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                last_gesture_time = current_time

        if index_and_middle_together and index_finger_up and middle_finger_up:
            if current_time - last_gesture_time > 0.3:
                mouse.click(Button.left, 2)
                cv2.putText(frame, "Double Click", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                last_gesture_time = current_time

def main():
    global last_click_time, click_count, last_gesture_time
    last_click_time = 0
    click_count = 0
    last_gesture_time = 0
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)
            landmarks = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm)
            detect_gesture(frame, landmarks, processed)
            time.sleep(0.01)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()