import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class AIEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def evaluate(self, test_loader):
        self.model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for batch in test_loader:
                inputs, batch_labels = batch
                inputs, batch_labels = inputs.to(self.device), batch_labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions)
        matrix = confusion_matrix(labels, predictions)
        return accuracy, report, matrix

    def get_feature_importances(self):
        importances = []
        for param in self.model.parameters():
            if param.requires_grad:
                importances.append(param.data.cpu().numpy().flatten())
        importances = np.concatenate(importances)
        return importances

def evaluate_ai_model(model, device, test_loader):
    evaluator = AIEvaluator(model, device)
    accuracy, report, matrix = evaluator.evaluate(test_loader)
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Classification Report:\n{report}')
    print(f'Confusion Matrix:\n{matrix}')
    importances = evaluator.get_feature_importances()
    return accuracy, report, matrix, importances
