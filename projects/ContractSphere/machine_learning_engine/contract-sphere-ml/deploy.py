import torch
import torch.nn as nn
from contract_classifier import ContractClassifier

def load_model(model_path):
    model = ContractClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def deploy_model(model, input_data):
    with torch.no_grad():
        output = model(torch.tensor(input_data, dtype=torch.float))
        _, predicted = torch.max(output, 1)
        return predicted.item()

def get_contract_label(predicted_label):
    labels = {
        0: 'Contract A',
        1: 'Contract B',
        2: 'Contract C',
        3: 'Contract D',
        4: 'Contract E',
        5: 'Contract F',
        6: 'Contract G',
        7: 'Contract H'
    }
    return labels[predicted_label]

def main():
    model_path = 'models/contract_classifier.pth'
    input_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
    model = load_model(model_path)
    predicted_label = deploy_model(model, input_data)
    contract_label = get_contract_label(predicted_label)
    print(f'Predicted contract label: {contract_label}')

if __name__ == '__main__':
    main()
