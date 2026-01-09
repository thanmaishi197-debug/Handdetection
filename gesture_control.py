import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions
import os

MODEL_PATH = "hand_landmarker.task"

detector = None
if os.path.exists(MODEL_PATH):
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=1,
        running_mode=RunningMode.IMAGE
    )
    detector = HandLandmarker.create_from_options(options)


def detect_gesture(frame):
    if detector is None:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]

    h, w, _ = frame.shape
    lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    tip_ids = [4, 8, 12, 16, 20]
    fingers_up = 0

    if lm_list[4][1] < lm_list[3][1]:
        fingers_up += 1

    for i in range(1, 5):
        if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i] - 2][1]:
            fingers_up += 1

    return f"{fingers_up}_fingers"
