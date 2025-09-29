import cv2


class Camera:
    def __init__(self, index=0, width=1280, height=720):
        self.cam = cv2.VideoCapture(index)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def capture(self):
        success, frame = self.cam.read()
        if not success:
            return None
        return frame

    def release(self):
        if self.cam:
            self.cam.release()