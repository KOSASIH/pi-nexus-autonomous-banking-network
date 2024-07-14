# File name: advanced_nlp.py
import torch
import transformers

# Load pre-trained language model
model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Define custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        encoding = model.encode_plus(text, 
                                      add_special_tokens=True, 
                                      max_length=512, 
                                      return_attention_mask=True, 
                                      return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
        }

# Load dataset
data = [...]; // Load dataset
labels = [...]; // Load labels

# Create custom dataset instance
dataset = CustomDataset(data, labels)

# Create data loader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')
    model.eval()
