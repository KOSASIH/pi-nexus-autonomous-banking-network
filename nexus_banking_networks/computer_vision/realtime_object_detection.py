import cv2
import numpy as np

# Load TensorFlow Lite model
interpreter = cv2.dnn.readNetFromTensorflow('model.tflite')

def detect_objects(image):
    # Preprocess image
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (0, 0, 0), swapRB=True, crop=False)

    # Run object detection
    interpreter.setInput(blob)
    interpreter.forward()

    # Extract bounding boxes and class probabilities
    boxes = []
    class_probs = []
    for i in range(interpreter.getUnconnectedOutLayers()[0].getOutputSize()):
        class_id = int(interpreter.getUnconnectedOutLayers()[0].getOutputs()[i][0])
        confidence = interpreter.getUnconnectedOutLayers()[0].getOutputs()[i][1]
        if confidence > 0.5:
            center_x = int(interpreter.getUnconnectedOutLayers()[0].getOutputs()[i][3] * width)
            center_y = int(interpreter.getUnconnectedOutLayers()[0].getOutputs()[i][4] * height)
            w = int(interpreter.getUnconnectedOutLayers()[0].getOutputs()[i][5] * width)
            h = int(interpreter.getUnconnectedOutLayers()[0].getOutputs()[i][6] * height)
            boxes.append((center_x, center_y, w, h))
            class_probs.append(confidence)

    return boxes, class_probs

# Example usage
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    boxes, class_probs = detect_objects(frame)

    # Draw bounding boxes
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)

    cv2.imshow('Real-time Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
