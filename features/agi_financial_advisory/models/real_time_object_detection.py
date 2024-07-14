# File name: real_time_object_detection.py
import cv2
import numpy as np
from tensorflow.keras.applications import YOLOv3

# Load YOLOv3 model
model = YOLOv3(weights='yolov3.weights')

# Load video capture device
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    frame = cv2.resize(frame, (416, 416))
    frame = frame / 255.0

    # Run object detection
    outputs = model.predict(frame)

    # Draw bounding boxes
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                x, y, w, h = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    # Display output
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
