import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import torch
from torch.nn import functional as F

class AttentionExplainableAnomalyDetector(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionExplainableAnomalyDetector, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.attention = torch.nn.MultiHeadAttention(hidden_dim, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.attention(x, x)
        x = self.fc2(x)
        return x

class AttentionLayer(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(AttentionLayer, self).__init__()
        self.query_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)
        attention_weights = torch.matmul(query, key.T) / math.sqrt(hidden_dim)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = attention_weights * value
        return output

# Example usage
data = pd.read_csv('data.csv')
model = AttentionExplainableAnomalyDetector(input_dim=data.shape[1], hidden_dim=128, output_dim=1)
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
