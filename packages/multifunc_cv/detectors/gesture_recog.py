import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

VisionRunningMode = mp.tasks.vision.RunningMode

class GestureRecog:
    def __init__(self, model_path, num_hands=2, min_conf=0.6):
        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf)
        self.detector = vision.GestureRecognizer.create_from_options(options)
        self.t0 = time.perf_counter()
        self.last_ts = -1

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts = int((time.perf_counter() - self.t0) * 1000)
        if ts <= self.last_ts:
            ts = self.last_ts + 1
        self.last_ts = ts
        gestures = []
        gesture_result = self.detector.recognize_for_video(mp_image, ts)
        if gesture_result:
            for i, landmarks in enumerate(gesture_result.hand_landmarks or []):
                label = None
                score = None
                if gesture_result.gestures and len(gesture_result.gestures) > i and gesture_result.gestures[i]:
                    top = gesture_result.gestures[i][0]
                    label = top.category_name
                    score = float(top.score)
                gestures.append({
                    "landmarks": [(int(l.x*frame.shape[1]), int(l.y*frame.shape[0])) for l in landmarks],
                    "gesture": label,
                    "score": score
                })
        return gestures

    def close(self):
        del self.detector