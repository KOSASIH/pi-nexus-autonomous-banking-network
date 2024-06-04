# biometric_auth.py
import face_recognition

class BiometricAuth:
    def __init__(self, face_recognition_model):
        self.face_recognition_model = face_recognition_model

    def authenticate(self, face_image):
        # validate face image
        face_encoding = face_recognition.face_encodings(face_image)[0]
        return self.face_recognition_model.compare_faces(face_encoding)
