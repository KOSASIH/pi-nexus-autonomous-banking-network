# sidra_chain_vision.py
import cv2
import numpy as np

class SidraChainVision:
    def __init__(self):
        # Initialize camera
        self.camera = cv2.VideoCapture(0)

    def capture_image(self):
        # Capture an image from the camera
        ret, frame = self.camera.read()
        return frame

    def process_image(self, frame):
        # Process the image using OpenCV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges

    def detect_objects(self, edges):
        # Detect objects in the image using OpenCV
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                objects.append(contour)
        return objects
