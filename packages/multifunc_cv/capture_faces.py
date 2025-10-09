import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
from camera import Camera
from detectors.face_detect import FaceDetect
from draw import draw_face
import time

FACE_MODEL = "models/blaze_face_short_range.tflite"

cam = Camera()
face = FaceDetect(FACE_MODEL)

try:
    while True:
        frame = cam.capture()
        if frame is None:
            break
        faces = face.detect(frame)
        draw_face(frame, faces)

        cv2.imshow("Face Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            timestamp = int(time.time())
            if faces:
                bbox = faces[0]['bbox']
                x, y, w, h = bbox
                cropped = frame[y:y + h, x:x + w]
                cv2.imwrite(f"data/my_face/face_{timestamp}.jpg", cropped)
            else:
                print("No face detected")

        if key == 27:
            break

finally:
    cam.release()
    face.close()
    cv2.destroyAllWindows()