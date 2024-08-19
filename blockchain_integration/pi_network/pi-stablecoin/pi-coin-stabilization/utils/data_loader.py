import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer

class PiDataset(Dataset):
    def __init__(self, data, target, tokenizer, max_len):
        self.data = data
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.target.iloc[idx]

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
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(file_path, target_variable, test_size=0.2, random_state=42):
    data = pd.read_csv(file_path)
    X = data.drop([target_variable], axis=1)
    y = data[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def create_data_loader(X, y, tokenizer, max_len, batch_size):
    dataset = PiDataset(X, y, tokenizer, max_len)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)
