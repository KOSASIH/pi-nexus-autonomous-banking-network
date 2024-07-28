import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Perform data preprocessing tasks (e.g., handle missing values, encode categorical variables)
    return data

def split_data(data, test_size=0.2):
    # Split data into training and testing sets
    return train_data, test_data
