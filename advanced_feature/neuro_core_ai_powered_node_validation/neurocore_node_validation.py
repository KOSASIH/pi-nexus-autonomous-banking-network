import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class NeuroCoreNodeValidation(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuroCoreNodeValidation, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) 
        self.fc3 = nn.Linear(hidden_dim, output_dim) 

    def forward(self, x):
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x)) 
        x = self.fc3(x) 
        return x

class NodeValidationDataset(Dataset):
    def __init__(self, node_data, labels):
        self.node_data = node_data
        self.labels = labels

    def __len__(self):
        return len(self.node_data)

    def __getitem__(self, idx):
        node_data = self.node_data.iloc[idx, :]
        label = self.labels.iloc[idx]
        return {
            'node_data': torch.tensor(node_data.values, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

def generate_node_data(node_id, node_type, node_ip, node_port, node_os, node_arch):
    # Generate node data features
    node_data = {
        'node_id': node_id,
        'node_type': node_type,
        'node_ip': node_ip,
        'node_port': node_port,
        'node_os': node_os,
        'node_arch': node_arch,
        'node_uptime': np.random.uniform(0, 100), 
        'node_cpu_usage': np.random.uniform(0, 100), 
        'node_memory_usage': np.random.uniform(0, 100), 
        'node_disk_usage': np.random.uniform(0, 100), 
    }
    return pd.DataFrame([node_data])

def train_neurocore_model(node_data, labels, epochs=100, batch_size=32):
    # Create dataset and data loader
    dataset = NodeValidationDataset(node_data, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize NeuroCore model
    model = NeuroCoreNodeValidation(input_dim=8, hidden_dim=16, output_dim=2)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train NeuroCore model
    for epoch in range(epochs):
        for batch in data_loader:
            node_data = batch['node_data'].to(device)
            label = batch['label'].to(device)
            optimizer.zero_grad()
            output = model(node_data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    return model

def validate_node(node_data, model):
    # Validate node using NeuroCore model
    node_data = torch.tensor(node_data.values, dtype=torch.float)
    output = model(node_data)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# Generate node data
node_data = generate_node_data("node1", "validator", "192.168.1.100", 8080, "Linux", "x86_64")

# Generate labels
labels = pd.DataFrame([{'label': 1}]) 

# Train NeuroCore model
model = train_neurocore_model(node_data, labels)

# Validate node
node_validation_result = validate_node(node_data, model)
print(f'Node validation result: {node_validation_result}')
