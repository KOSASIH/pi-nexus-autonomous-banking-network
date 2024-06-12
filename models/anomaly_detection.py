import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import IsolationForest

class AnomalyDetectionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AnomalyDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AnomalyDetectionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

def train_anomaly_detection_model(data, labels, epochs=100, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnomalyDetectionModel(input_dim=data.shape[1], hidden_dim=128)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = AnomalyDetectionDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
    return model

def detect_anomalies(model, data):
    outputs = model(data)
    anomalies = outputs > 0.5
    return anomalies

def visualize_anomalies(data, anomalies):
    import matplotlib.pyplot as plt
    plt.scatter(data[:, 0], data[:, 1], c=anomalies)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Anomaly Detection")
    plt.show()
