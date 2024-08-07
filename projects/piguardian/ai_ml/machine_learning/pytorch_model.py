import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class PyTorchModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(PyTorchModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

class PyTorchDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return {'X': torch.tensor(X), 'y': torch.tensor(y)}

class PyTorchTrainer:
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

    def train(self, dataset, batch_size=32, epochs=10):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for batch in data_loader:
                X = batch['X'].to(self.device)
                y = batch['y'].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

    def evaluate(self, dataset):
        self.model.eval()
        total_correct = 0
        with torch.no_grad():
            for batch in dataset:
                X = batch['X'].to(self.device)
                y = batch['y'].to(self.device)
                outputs = self.model(X)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == y).sum().item()
        accuracy = total_correct / len(dataset)
        return accuracy

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
