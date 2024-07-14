# File name: advanced_computer_vision.py
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class DefectDetector:
    def __init__(self, model_path):
        self.model = cv2.dnn.readNetFromDarknet(model_path, "yolov3.cfg")
        self.classifier = RandomForestClassifier(n_estimators=100)

    def detect_defects(self, image):
        height, width, channels = image.shape
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.model.setInput(blob)
        outs = self.model.forward(self.model.getUnconnectedOutLayersNames())
        defects = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    defect_type = self.classifier.predict(np.array([detection[4:]]))[0]
                    defects.append((x, y, w, h, defect_type))
        return defects

defect_detector = DefectDetector("defect_detection_model.weights")
image = cv2.imread("image.jpg")
defects = defect_detector.detect_defects(image)
for defect in defects:
    x, y, w, h, defect_type = defect
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, defect_type, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
cv2.imshow("Defect Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
