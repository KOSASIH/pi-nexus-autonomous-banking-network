import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from node_optimizer_model import NodeOptimizer
from node_optimizer_dataset import NodeOptimizerDataset
from utils import *

class NodeOptimizerTrainer:
    def __init__(self, model, dataset, batch_size, epochs, learning_rate, device):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            for batch in data_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    def evaluate(self):
        self.model.eval()
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        print(f'Evaluation Loss: {avg_loss}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = NodeOptimizerDataset('data/node_performance_data.csv')
    model = NodeOptimizer(input_dim=6, hidden_dim=128, output_dim=1)
    trainer = NodeOptimizerTrainer(model, dataset, batch_size=32, epochs=100, learning_rate=0.001, device=device)
    trainer.train()
    trainer.evaluate()
    trainer.save_model('models/node_optimizer.pth')
