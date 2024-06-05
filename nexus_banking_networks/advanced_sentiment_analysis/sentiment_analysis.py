# sentiment_analysis.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SentimentAnalysis(nn.Module):
  def __init__(self):
    super(SentimentAnalysis, self).__init__()
    self.fc1 = nn.Linear(128, 64)
    self.fc2 = nn.Linear(64, 2)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x

class SentimentDataset(Dataset):
  def __init__(self, data, labels):
    self.data = data
    self.labels = labels

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]

# Example usage:
model = SentimentAnalysis()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

data = # load customer feedback data
labels = # load sentiment labels
dataset = SentimentDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(10):
  for batch in dataloader:
    inputs, labels = batch
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print("Epoch:", epoch, "Loss:", loss.item())
