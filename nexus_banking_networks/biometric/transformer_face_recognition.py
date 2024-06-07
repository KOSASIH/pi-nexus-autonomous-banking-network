import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class TransformerFaceRecognizer(nn.Module):
    def __init__(self, num_classes):
        super(TransformerFaceRecognizer, self).__init__()
        self.transformer = models.Transformer(num_layers=6, hidden_size=256, num_heads=8)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)))
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

# Example usage
recognizer = TransformerFaceRecognizer(num_classes=10)
image_path = 'image.jpg'
img = cv2.imread(image_path)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)
img = torch.from_numpy(img)
prediction = recognizer(img)
print(f'Prediction: {prediction}')
