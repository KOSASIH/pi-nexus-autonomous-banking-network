import cv2

def detect_objects(image_path):
    image = cv2.imread(image_path)
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def detect_edges(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges
