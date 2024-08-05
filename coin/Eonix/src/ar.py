# ar.py
import cv2
import numpy as np
from PIL import Image

class EonixAR:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)

    def capture_image(self):
        ret, frame = self.camera.read()
        return frame

    def detect_markers(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        markers = cv2.aruco.detectMarkers(gray)
        return markers

    def track_markers(self, markers):
        for marker in markers:
            corners = marker[0]
            cv2.drawContours(image, [corners], -1, (0, 255, 0), 2)
        return image

    def display_ar(self, image):
        cv2.imshow('AR', image)
        cv2.waitKey(1)
