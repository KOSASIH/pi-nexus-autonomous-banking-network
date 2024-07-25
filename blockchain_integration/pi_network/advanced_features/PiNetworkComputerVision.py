# Importing necessary libraries
import cv2
import numpy as np

# Class for computer vision
class PiNetworkComputerVision:
    def __init__(self):
        self.net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

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
cv = PiNetworkComputerVision()
image = cv2.imread("image.jpg")
outs = cv.detect_objects(image)
print(outs)
