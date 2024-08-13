import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class AnomalyDetection:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.tsne = TSNE(n_components=2, random_state=42)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, data):
        # Data preprocessing
        data = self.scaler.fit_transform(data)
        data = self.pca.fit_transform(data)
        data = self.tsne.fit_transform(data)

        # Model selection
        if self.config['model'] == 'isolation_forest':
            self.model = IsolationForest(contamination=0.1, random_state=42)
        elif self.config['model'] == 'one_class_svm':
            self.model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
        elif self.config['model'] == 'local_outlier_factor':
            self.model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        elif self.config['model'] == 'autoencoder':
            self.model = Autoencoder(data.shape[1], 128, 64)
            self.model.to(self.device)
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Model training
        if self.config['model'] != 'autoencoder':
            self.model.fit(data)
        else:
            dataset = AnomalyDetectionDataset(data)
            data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            for epoch in range(100):
                for batch in data_loader:
                    batch = batch.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(batch)
                    loss = self.criterion(outputs, batch)
                    loss.backward()
                    self.optimizer.step()
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    def predict(self, data):
        # Data preprocessing
        data = self.scaler.transform(data)
        data = self.pca.transform(data)
        data = self.tsne.transform(data)

        # Model prediction
        if self.config['model'] != 'autoencoder':
            predictions = self.model.predict(data)
        else:
            dataset = AnomalyDetectionDataset(data)
            data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
            predictions = []
            with torch.no_grad():
                for batch in data_loader:
                    batch = batch.to(self.device)
                    outputs = self.model(batch)
                    predictions.extend(torch.sigmoid(outputs).cpu().numpy())

        return predictions

    def evaluate(self, data, labels):
        predictions = self.predict(data)
        precision, recall, _ = precision_recall_curve(labels, predictions)
        auc_score = auc(recall, precision)
        accuracy = accuracy_score(labels, predictions > 0.5)
        print(f'AUC-PR: {auc_score:.4f}, Accuracy: {accuracy:.4f}')

class AnomalyDetectionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

if __name__ == '__main__':
    config = {
        'model': 'autoencoder',
        'data_path': 'data.csv'
    }
    data = pd.read_csv(config['data_path'])
    anomaly_detection = AnomalyDetection(config)
    anomaly_detection.fit(data)
    anomaly_detection.evaluate(data, np.zeros(len(data))) 
