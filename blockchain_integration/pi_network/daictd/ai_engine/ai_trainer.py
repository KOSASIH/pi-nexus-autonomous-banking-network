import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from ai_model import AIModel, CNNModel, LSTMModel

class AITrainer:
    def __init__(self, model, device, criterion, optimizer):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, train_loader, epochs):
        for epoch in range(epochs):
            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    def evaluate(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        accuracy = correct / len(test_loader.dataset)
        print(f'Test Loss: {test_loss / len(test_loader)}')
        print(f'Test Accuracy: {accuracy:.2f}%')

def train_ai_model(model_type, input_dim, hidden_dim, output_dim, device, criterion, optimizer, train_loader, epochs):
    if model_type == 'fc':
        model = AIModel(input_dim, hidden_dim, output_dim)
    elif model_type == 'cnn':
        model = CNNModel(input_dim, hidden_dim, output_dim)
    elif model_type == 'lstm':
        model = LSTMModel(input_dim, hidden_dim, output_dim)
    trainer = AITrainer(model, device, criterion, optimizer)
    trainer.train(train_loader, epochs)
    return trainer

def evaluate_ai_model(trainer, test_loader):
    trainer.evaluate(test_loader)
