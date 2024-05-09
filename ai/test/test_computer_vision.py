import unittest
import cv2
from ai.computer_vision import detect_objects, detect_edges

class TestComputerVision(unittest.TestCase):
    def test_detect_objects(self):
        image_path = 'test_image.jpg'
        faces = detect_objects(image_path)
        self.assertIsInstance(faces, list)
self.assertGreaterEqual(len(faces), 0)

    def test_detect_edges(self):
        image_path = 'test_image.jpg'
        edges = detect_edges(image_path)
        self.assertIsInstance(edges, np.ndarray)

if __name__ == '__main__':
    unittest.main()
