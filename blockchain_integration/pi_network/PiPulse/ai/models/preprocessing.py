import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    # Handle missing values
    data.fillna(data.mean(), inplace=True)

    # Scale data using Min-Max Scaler
    scaler = MinMaxScaler()
    data[['cpu_usage', 'memory_usage', 'disk_usage']] = scaler.fit_transform(data[['cpu_usage', 'memory_usage', 'disk_usage']])

    # Extract features and labels
    features = data.drop(['label'], axis=1)
    labels = data['label']

    return features, labels

def split_data(features, labels, test_size=0.2):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
