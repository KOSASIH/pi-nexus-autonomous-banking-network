# Importing necessary libraries
import cv2
import numpy as np

# Class for computer vision
class PiNetworkComputerVision2:
    def __init__(self):
        self.net = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4.weights")

    # Function to detect objects
    def detect_objects(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), [0,0,0], 1, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.getOutputsNames(self.net))
        return outs

    # Function to get output names
    def getOutputsNames(self, net):
        layersNames = net.getLayerNames()
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Example usage
cv2 = PiNetworkComputerVision2()
image = cv2.imread("image.jpg")
outs = cv2.detect_objects(image)
print(outs)
