import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class AnomalyDetectionModel(nn.Module):
    def __init__(self, num_features):
        super(AnomalyDetectionModel, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AnomalyDetectionSystem:
    def __init__(self, anomaly_detection_model):
        self.anomaly_detection_model = anomaly_detection_model

    def detect_anomalies(self, data):
        data_loader = DataLoader(data, batch_size=32, shuffle=True)
        anomalies = []
        for batch in data_loader:
            input_data, _ = batch
            output = self.anomaly_detection_model(input_data)
            anomalies.extend(torch.sigmoid(output).detach().numpy())
        return anomalies
