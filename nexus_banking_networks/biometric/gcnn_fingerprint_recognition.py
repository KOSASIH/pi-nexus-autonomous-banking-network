import cv2
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data

class GCNNFingerprintRecognizer(nn.Module):
    def __init__(self, num_classes):
        super(GCNNFingerprintRecognizer, self).__init__()
        self.conv1 = pyg_nn.GraphConv(16, 32)
        self.conv2 = pyg_nn.GraphConv(32, 64)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x

# Example usage
recognizer = GCNNFingerprintRecognizer(num_classes=10)
image_path = 'fingerprint_image.jpg'
img = cv2.imread(image_path)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)
img = torch.from_numpy(img)
data = pyg_data.Data(x=img, edge_index=torch.tensor([[0, 1], [1, 2], [2, 0]]))
prediction = recognizer(data)
print(f'Prediction: {prediction}')
