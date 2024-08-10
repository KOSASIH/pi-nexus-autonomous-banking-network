import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum')
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == target).sum().item()

    accuracy = total_correct / len(data_loader.dataset)
    loss = total_loss / len(data_loader)
    return accuracy, loss

def classification_report_to_df(report):
    report_df = pd.DataFrame(report).transpose()
    report_df.columns = ['precision', 'recall', 'f1-score', 'support']
    return report_df

def evaluate_model_with_report(model, data_loader, device):
    accuracy, loss = evaluate_model(model, data_loader, device)
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(target.cpu().numpy())

    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = classification_report_to_df(report)
    return accuracy, loss, report_df
