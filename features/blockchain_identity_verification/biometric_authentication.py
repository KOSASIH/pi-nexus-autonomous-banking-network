# File name: biometric_authentication.py
import face_recognition
import numpy as np

class BiometricAuthentication:
    def __init__(self):
        self.known_faces = []

    def enroll_face(self, image):
        face_encoding = face_recognition.face_encodings(image)[0]
        self.known_faces.append(face_encoding)

    def authenticate(self, image):
        unknown_face_encoding = face_recognition.face_encodings(image)[0]
        results = face_recognition.compare_faces(self.known_faces, unknown_face_encoding)
        return results
