import torch
import torch.nn as nn
import torch.optim as optim

class AGIRiskAssessment(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AGIRiskAssessment, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train(self, dataset, epochs=100):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(dataset)
            loss = criterion(outputs, dataset)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    def assess_risk(self, input_data):
        output = self(input_data)
        return output.detach().numpy()
