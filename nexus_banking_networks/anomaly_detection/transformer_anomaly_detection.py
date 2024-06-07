import pandas as pd
import numpy as np
import torch
from torch.nn import Transformer
from torch.nn import functional as F

class TransformerAnomalyDetector(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(TransformerAnomalyDetector, self).__init__()
        self.encoder = TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.decoder = TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.fc(x)
        return x

class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = torch.nn.MultiHeadAttention(d_model, nhead)
        self.feed_forward = torch.nn.Linear(d_model, dim_feedforward)

    def forward(self, x):
        x = self.self_attn(x, x)
        x = F.relu(self.feed_forward(x))
        return x

class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = torch.nn.MultiHeadAttention(d_model, nhead)
        self.feed_forward = torch.nn.Linear(d_model, dim_feedforward)

    def forward(self, x):
        x = self.self_attn(x, x)
        x = F.relu(self.feed_forward(x))
        return x

# Example usage
data = pd.read_csv('data.csv')
model = TransformerAnomalyDetector(input_dim=data.shape[1], hidden_dim=128, output_dim=1, num_heads=8)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(torch.tensor(data.values))
    loss = criterion(output, torch.tensor([0.0]))
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Anomaly detection
anomaly_scores = []
for i in range(len(data)):
    output = model(torch.tensor(data.iloc[i:i+100].values))
    anomaly_scores.append(output.item())

anomaly_threshold = 3
anomalies = [i for i, score in enumerate(anomaly_scores) if score > anomaly_threshold]
print(f'Anomalies detected: {anomalies}')
