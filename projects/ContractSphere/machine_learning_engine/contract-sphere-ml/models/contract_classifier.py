import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class ContractClassifier(nn.Module):
    def __init__(self):
        super(ContractClassifier, self).__init__()
        self.fc1 = nn.Linear(128, 64)  # input layer (128) -> hidden layer (64)
        self.fc2 = nn.Linear(64, 32)  # hidden layer (64) -> hidden layer (32)
        self.fc3 = nn.Linear(32, 8)  # hidden layer (32) -> output layer (8)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # activation function for hidden layer
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ContractDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx, :]
        label = self.labels.iloc[idx]
        return {
            'data': torch.tensor(data.values, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data(file_path):
    data = pd.read_csv(file_path)
    labels = data['label']
    data.drop('label', axis=1, inplace=True)
    return data, labels

def train_model(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in train_loader:
        data, label = batch['data'].to(device), batch['label'].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            data, label = batch['data'].to(device), batch['label'].to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == label).sum().item()
    return correct / len(test_loader)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data, labels = load_data('data/train.csv')
    dataset = ContractDataset(data, labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = ContractClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        loss = train_model(model, device, train_loader, optimizer, criterion)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
    test_data, test_labels = load_data('data/test.csv')
    test_dataset = ContractDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    accuracy = evaluate_model(model, device, test_loader)
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()
