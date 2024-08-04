import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        data = pd.read_csv(self.data_path)
        return data

    def preprocess_data(self, data):
        scaler = StandardScaler()
        data[['feature1', 'feature2',...]] = scaler.fit_transform(data[['feature1', 'feature2',...]])
        return data

    def split_data(self, data):
        train_data, test_data = data.split(test_size=0.2, random_state=42)
        return train_data, test_data

    def save_data(self, data, path):
        data.to_csv(path, index=False)
