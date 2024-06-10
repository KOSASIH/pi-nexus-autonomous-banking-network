import face_recognition
import cv2

class BiometricAuth:
    def __init__(self):
        self.known_faces = []
        self.known_face_names = []

   def add_face(self, image, name):
        face_image = face_recognition.load_image_file(image)
        face_encoding = face_recognition.face_encodings(face_image)[0]
        self.known_faces.append(face_encoding)
        self.known_face_names.append(name)

    def authenticate(self, image):
        face_image = face_recognition.load_image_file(image)
        face_encoding = face_recognition.face_encodings(face_image)[0]
        results = face_recognition.compare_faces(self.known_faces, face_encoding)
        if True in results:
            index = results.index(True)
            return self.known_face_names[index]
        else:
            return None

# Example usage:
auth = BiometricAuth()
auth.add_face("user1.jpg", "User 1")
auth.add_face("user2.jpg", "User 2")
authenticated_user = auth.authenticate("test_image.jpg")
print(authenticated_user)
