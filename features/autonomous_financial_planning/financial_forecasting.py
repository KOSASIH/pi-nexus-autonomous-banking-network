# File name: financial_forecasting.py
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data

class FinancialForecastingModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(FinancialForecastingModel, self).__init__()
        self.conv1 = pyg_nn.GraphConv(num_features, 128)
        self.conv2 = pyg_nn.GraphConv(128, 128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = FinancialForecastingModel(num_features=10, num_classes=1)
data = pyg_data.Data(x=torch.randn(100, 10), edge_index=torch.tensor([[0, 1], [1, 2]]))
output = model(data)
print(output)
