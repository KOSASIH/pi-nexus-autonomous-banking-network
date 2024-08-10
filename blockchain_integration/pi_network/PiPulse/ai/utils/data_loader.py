import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class PiPulseDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx, :]
        label = self.labels.iloc[idx]

        if self.transform:
            data = self.transform(data)

        return {
            'data': torch.tensor(data, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

class PiPulseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=4):
        super(PiPulseDataLoader, self).__init__(dataset, batch_size, shuffle, num_workers)

    def collate_fn(self, batch):
        data = torch.cat([item['data'] for item in batch], dim=0)
        labels = torch.cat([item['label'] for item in batch], dim=0)
        return data, labels

def load_data(file_path, batch_size, shuffle=True, num_workers=4):
    data = pd.read_csv(file_path)
    labels = data['label']
    data.drop('label', axis=1, inplace=True)

    scaler = StandardScaler()
    data[['cpu_usage', 'memory_usage', 'disk_usage']] = scaler.fit_transform(data[['cpu_usage', 'memory_usage', 'disk_usage']])

    dataset = PiPulseDataset(data, labels)
    data_loader = PiPulseDataLoader(dataset, batch_size, shuffle, num_workers)

    return data_loader
