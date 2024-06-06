import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class NLPChatbot:
    def __init__(self, vocab_size, hidden_size, output_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model= nn.Sequential(
            nn.Embedding(vocab_size, hidden_size),
            nn.LSTM(hidden_size, hidden_size),
            nn.Linear(hidden_size, output_size)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_model(self, dataset):
        for epoch in range(100):
            for input, target in dataset:
                input = torch.tensor(input)
                target = torch.tensor(target)
                self.optimizer.zero_grad()
                output = self.model(input)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

    def generate_response(self, input):
        input = torch.tensor(input)
        output = self.model(input)
        return output

# Example usage:
nlp_chatbot = NLPChatbot(1000, 128, 128)
dataset = [(['hello', 'world'], ['hi', 'there'])]
nlp_chatbot.train_model(dataset)

# Generate a response
input = ['hello', 'world']
response = nlp_chatbot.generate_response(input)
print(f'Response: {response}')
