import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class AI_Agent(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(AI_Agent, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        outputs = self.fc(pooled_output)
        return outputs

    def train(self, dataset, epochs=10, batch_size=32):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        for epoch in range(epochs):
            for batch in dataset:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.forward(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    def predict(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        attention_mask = self.tokenizer.encode(input_text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        outputs = self.forward(input_ids, attention_mask)
        return torch.argmax(outputs)

# Example usage:
agent = AI_Agent(num_classes=8)
agent.train(dataset)
print(agent.predict('This is a sample text'))
