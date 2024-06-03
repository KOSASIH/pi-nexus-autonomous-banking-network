import face_recognition
import cv2

class BiometricAuth:
    def __init__(self):
        self.known_faces = []
        self.known_face_names = []

    def enroll_face(self, face_image, face_name):
        face_encoding = face_recognition.face_encodings(face_image)[0]
        self.known_faces.append(face_encoding)
        self.known_face_names.append(face_name)

    def authenticate_face(self, face_image):
        face_locations = face_recognition.face_locations(face_image)
        face_encodings = face_recognition.face_encodings(face_image, face_locations)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_faces, face_encoding)
            if True in matches:
                return self.known_face_names[matches.index(True)]
        return None

    def capture_face_image(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return frame

# Example usage:
auth = BiometricAuth()
face_image = auth.capture_face_image()
auth.enroll_face(face_image, "John Doe")

face_image = auth.capture_face_image()
authenticated_name = auth.authenticate_face(face_image)
if authenticated_name:
    print(f"Authenticated as {authenticated_name}!")
else:
    print("Authentication failed!")
