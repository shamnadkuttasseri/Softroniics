import cv2
from detector import FaceHandDetector

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)  # 0 for webcam
        self.detector = FaceHandDetector()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if success:
            frame = self.detector.detect(frame)
            _, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
