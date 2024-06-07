import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class YOLOv3(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv3, self).__init__()
        self.model = models.detection.yolo.YOLOv3(num_classes)

    def forward(self, x):
        return self.model(x)

class ObjectDetectionSystem:
    def __init__(self, yolov3_model):
        self.yolov3_model = yolov3_model
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect_objects(self, image):
        image = self.transform(image)
        image = image.unsqueeze(0)
        output = self.yolov3_model(image)
        return output
