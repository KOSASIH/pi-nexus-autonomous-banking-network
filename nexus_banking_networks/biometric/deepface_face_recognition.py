import cv2
import numpy as np
from deepface import DeepFace

class DeepFaceFaceRecognizer:
    def __init__(self, model_name='VGG-Face'):
        self.model = DeepFace.build_model(model_name)

    def recognize_face(self, image_path):
        img = cv2.imread(image_path)
        face_detected = DeepFace.detect_face(img)
        if face_detected:
            face_representation = DeepFace.represent(img, model_name=self.model_name)
            return face_representation
        else:
            return None

    def verify_faces(self, image_path1, image_path2):
        face_rep1 = self.recognize_face(image_path1)
        face_rep2 = self.recognize_face(image_path2)
        if face_rep1 is not None and face_rep2 is not None:
            distance = DeepFace.verify(face_rep1, face_rep2)
            return distance
        else:
            return None

# Example usage
recognizer = DeepFaceFaceRecognizer()
image_path1 = 'image1.jpg'
image_path2 = 'image2.jpg'
distance = recognizer.verify_faces(image_path1, image_path2)
print(f'Distance: {distance}')
