# pi_nexus/computer_vision.py
import cv2

class ComputerVision:
    def __init__(self) -> None:
        self.capture = cv2.VideoCapture(0)

    def capture_image(self) -> np.ndarray:
        ret, frame = self.capture.read()
        return frame

    def process_image(self, image: np.ndarray) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return {'edges': edges}
