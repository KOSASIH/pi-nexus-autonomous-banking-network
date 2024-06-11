# nlp.py

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

class NLP:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()

    def load_data(self, data_file):
        with open(data_file, 'r') as f:
            data = json.load(f)
        return data

    def preprocess_data(self, data):
        inputs = []
        labels = []
        for item in data:
            text = item['text']
            label = item['label']
            inputs.append(self.tokenizer.encode_plus(text, 
                                                      add_special_tokens=True, 
                                                      max_length=512, 
                                                      return_attention_mask=True, 
                                                      return_tensors='pt'))
            labels.append(label)
        return inputs, labels

    def create_data_loader(self, inputs, labels, batch_size):
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return data_loader

    def train_model(self, data_loader, epochs, learning_rate):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in data_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')
        self.model.eval()

    def evaluate_model(self, data_loader):
        self.model.eval()
        total_correct = 0
        total_labels = []
        total_preds = []
        with torch.no_grad():
            for batch in data_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                _, preds = torch.max(logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_labels.extend(labels.cpu().numpy())
                total_preds.extend(preds.cpu().numpy())
        accuracy = accuracy_score(total_labels, total_preds)
        report = classification_report(total_labels, total_preds)
        matrix = confusion_matrix(total_labels, total_preds)
        return accuracy, report, matrix

    def save_model(self, model_file):
        torch.save(self.model.state_dict(), model_file)

    def load_model(self, model_file):
        self.model.load_state_dict(torch.load(model_file))

nlp = NLP('bert-base-uncased', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
