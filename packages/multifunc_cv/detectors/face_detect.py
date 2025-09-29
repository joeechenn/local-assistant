import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

VisionRunningMode = mp.tasks.vision.RunningMode

class FaceDetect:
    def __init__(self, model_path, min_conf=0.6):
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            min_detection_confidence=min_conf)
        self.detector = vision.FaceDetector.create_from_options(options)
        self.t0 = time.perf_counter()
        self.last_ts = -1

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts = int((time.perf_counter() - self.t0) * 1000)
        if ts <= self.last_ts:
            ts = self.last_ts + 1
        self.last_ts = ts
        faces = []
        face_result = self.detector.detect_for_video(mp_image, ts)
        if face_result.detections:
            for detection in face_result.detections:
                bbox = detection.bounding_box
                score = detection.categories[0].score if detection.categories else None
                faces.append({
                    'bbox': (bbox.origin_x, bbox.origin_y, bbox.width, bbox.height),
                    'score': score
                })
        return faces

    def close(self):
        del self.detector





