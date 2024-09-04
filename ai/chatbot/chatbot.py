import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class ChatbotDataset(Dataset):
    def __init__(self, intents, tokenizer, max_len):
        self.intents = intents
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.intents)

    def __getitem__(self, idx):
        intent = self.intents[idx]
        text = intent['text']
        label = intent['intent']

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class ChatbotModel(nn.Module):
    def __init__(self, bert_model, num_intents):
        super(ChatbotModel, self).__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert_model.config.hidden_size, num_intents)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)
        return outputs

class Chatbot:
    def __init__(self, intents_file, model_file, tokenizer_file, max_len=512):
        self.intents_file = intents_file
        self.model_file = model_file
        self.tokenizer_file = tokenizer_file
        self.max_len = max_len

        self.intents = self.load_intents()
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

    def load_intents(self):
        with open(self.intents_file, 'r') as f:
            intents = json.load(f)
        return intents

    def load_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.tokenizer_file)
        return tokenizer

    def load_model(self):
        model = ChatbotModel(BertModel.from_pretrained('bert-base-uncased'), len(self.intents))
        model.load_state_dict(torch.load(self.model_file, map_location=torch.device('cuda')))
        return model

    def train(self, batch_size=32, epochs=5):
        dataset = ChatbotDataset(self.intents, self.tokenizer, self.max_len)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

        self.model.eval()

    def predict(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten().to(device)
        attention_mask = encoding['attention_mask'].flatten().to(device)

        outputs = self.model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, dim=1)

        return predicted.item()

    def respond(self, text):
        intent_id = self.predict(text)
        intent = self.intents[intent_id]
        response = intent['response']
        return response

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

chatbot = Chatbot('intents.json', 'model.pth', 'tokenizer.json')
chatbot.train()

while True:
    text = input('User: ')
    response = chatbot.respond(text)
    print('Chatbot:', response)
