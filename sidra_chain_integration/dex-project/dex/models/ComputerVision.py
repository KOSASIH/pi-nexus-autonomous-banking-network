import cv2
import numpy as np

class ComputerVision:
    def __init__(self):
        self.detector = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

    def detect_objects(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), [0,0,0], 1, crop=False)
        self.detector.setInput(blob)
        outs = self.detector.forward(self.getOutputsNames())
        return outs

    def getOutputsNames(self):
        layersNames = self.detector.getLayerNames()
        return [layersNames[i[0] - 1] for i in self.detector.getUnconnectedOutLayers()]

    def draw_bounding_boxes(self, image, outs):
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > 0.5 and classId == 0:
                    center_x = int(detection[0] * image.shape[1])
                    center_y = int(detection[1] * image.shape[0])
                    w = int(detection[2] * image.shape[1])
                    h = int(detection[3] * image.shape[0])
                    x = center_x - w / 2
                    y = center_y - h / 2
                    cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
        return image
