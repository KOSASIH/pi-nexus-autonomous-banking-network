import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class PiPulseModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PiPulseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.scaler = StandardScaler()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def preprocess(self, data):
        data = self.scaler.fit_transform(data)
        return data

class PiPulseDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return {
            'data': torch.tensor(data, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

class PiPulseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True):
        super(PiPulseDataLoader, self).__init__(dataset, batch_size, shuffle)

    def collate_fn(self, batch):
        data = torch.cat([item['data'] for item in batch], dim=0)
        labels = torch.cat([item['label'] for item in batch], dim=0)
        return data, labels
