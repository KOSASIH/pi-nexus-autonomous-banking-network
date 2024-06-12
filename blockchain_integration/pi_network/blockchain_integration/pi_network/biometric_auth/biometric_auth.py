import face_recognition

class BiometricAuth:
    def __init__(self, face_recognition_model):
        self.face_recognition_model = face_recognition_model

    def authenticate(self, image):
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.face_recognition_model, face_encoding)
            if any(matches):
                return True
        return False

# Example usage:
face_recognition_model = face_recognition.load_image_file('known_faces.jpg')
biometric_auth = BiometricAuth(face_recognition_model)
image = face_recognition.load_image_file('new_face.jpg')
authenticated = biometric_auth.authenticate(image)
print(authenticated)
