import cv2
import numpy as np

class ComputerVision:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)

    def detect_edges(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)
        return edges

    def show_image(self, image):
        cv2.imshow('Edges', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    cv = ComputerVision('image.jpg')  # Load an image file
    edges = cv.detect_edges()
    cv.show_image(edges)
