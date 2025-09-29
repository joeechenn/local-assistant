import cv2
from camera import Camera
from detectors.face_detect import FaceDetect
from detectors.gesture_recog import GestureRecog
from draw import draw_face, draw_hands

FACE_MODEL = "models/blaze_face_short_range.tflite"
GESTURE_MODEL = "models/gesture_recognizer.task"

def main():
    cam = Camera(index=0, width=1280, height=720)
    face = FaceDetect(FACE_MODEL)
    gesture = GestureRecog(GESTURE_MODEL)

    try:
        while True:
            frame = cam.capture()
            if frame is None:
                break
            faces = face.detect(frame)
            draw_face(frame, faces)

            gestures = gesture.detect(frame)
            draw_hands(frame, gestures)

            cv2.imshow("Face and Gesture", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cam.release()
        face.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()