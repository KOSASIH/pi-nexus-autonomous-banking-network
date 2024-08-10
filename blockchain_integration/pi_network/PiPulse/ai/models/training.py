import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = 0
    with torch.set_grad_enabled(True):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()

    accuracy = correct / len(train_loader.dataset)
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')
    print(f'Training Accuracy: {accuracy:.4f}')

def evaluate_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()

    accuracy = correct / len(test_loader.dataset)
    print(f'Test Loss: {test_loss / len(test_loader)}')
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Classification Report:')
    print(classification_report(test_loader.dataset.labels, predicted.cpu().numpy()))
    print(f'Confusion Matrix:')
    print(confusion_matrix(test_loader.dataset.labels, predicted.cpu().numpy()))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PiPulseModel(input_dim=10, hidden_dim=20, output_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_data, test_data = split_data(features, labels, test_size=0.2)
    train_loader = PiPulseDataLoader(PiPulseDataset(train_data, labels), batch_size=32, shuffle=True)
    test_loader = PiPulseDataLoader(PiPulseDataset(test_data, labels), batch_size=32, shuffle=False)

    for epoch in range(10):
        train_model(model, device, train_loader, optimizer, criterion, epoch)
        evaluate_model(model, device, test_loader)

if __name__ == '__main__':
    main()
