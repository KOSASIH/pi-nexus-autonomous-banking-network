import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class FaceNet(nn.Module):
    def __init__(self, num_classes):
        super(FaceNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 2048)
        x = self.fc(x)
        return x

class FacialRecognitionSystem:
    def __init__(self, facenet_model):
        self.facenet_model = facenet_model
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def recognize_face(self, face_image):
        face_image = self.transform(face_image)
        face_image = face_image.unsqueeze(0)
        output = self.facenet_model(face_image)
        return output
