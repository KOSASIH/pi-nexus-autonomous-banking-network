import facenet

# Load FaceNet model
model = facenet.FaceNet()

# Load face embeddings
face_embeddings =...

# Verify a face
verified = model.verify(face_embeddings, "user-1")
