# dex_project_computer_vision.py
import cv2
import numpy as np

class DexProjectComputerVision:
    def __init__(self):
        pass

    def detect_objects(self, image):
        # Detect objects in an image using YOLO
        net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
        classes = []
        with open('coco.names', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        heights, widths = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * widths)
                    center_y = int(detection[1] * heights)
                    w = int(detection[2] * widths)
                    h = int(detection[3] * heights)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        return boxes, confidences, class_ids

    def track_objects(self, image, boxes, confidences, class_ids):
        # Track objects in an image using the Kalman filter
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        kf.measurementNoiseCov = np.array([[1, 0], [0, 1]])
        kf.errorCovPost = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        kf.statePost = np.array([[0, 0, 0, 0]]).T
        tracked_boxes = []
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            measurement = np.array([[x], [y]])
            kf.correct(measurement)
            prediction = kf.predict()
            tracked_boxes.append([int(prediction[0]), int(prediction[1]), w, h])
        return tracked_boxes

    def recognize_faces(self, image):
        # Recognize faces in an image using FaceNet
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
        face_descriptors = []
        for (x, y, w, h) in faces:
            face_image = image[y:y+h, x:x+w]
            face_descriptor = self.extract_face_descriptor(face_image)
            face_descriptors.append(face_descriptor)
        return face_descriptors

    def extract_face_descriptor(self, face_image):
        # Extract a face descriptor from a face image using FaceNet
        face_net = cv2.dnn.readNetFromTensorflow('facenet.pb')
        face_net.setInput(cv2.dnn.blobFromImage(face_image, 1/255, (160, 160), swapRB=False, crop=False))
        face_descriptor = face_net.forward()
        return face_descriptor
