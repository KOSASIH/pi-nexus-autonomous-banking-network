import cv2
import numpy as np
from tensorflow.keras.applications import YOLOv3
from tensorflow.keras.preprocessing.image import load_img

# Load YOLOv3 model
yolo = YOLOv3(weights='yolov3.h5')

def detect_objects(image_path):
    # Load image
    image = load_img(image_path)
    image_array = np.array(image)

    # Preprocess image
    image_array = cv2.resize(image_array, (416, 416))
    image_array = image_array / 255.0

    # Run object detection
    outputs = yolo.predict(image_array)
    detections = outputs[0]

    # Extract bounding boxes and class probabilities
    boxes = []
    class_probs = []
    for detection in detections:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            w = int(detection[2] * image.shape[1])
            h = int(detection[3] * image.shape[0])
            boxes.append((center_x, center_y, w, h))
            class_probs.append(confidence)

    return boxes, class_probs

# Example usage
image_path = 'path/to/image.jpg'
boxes, class_probs = detect_objects(image_path)
print(boxes, class_probs)
