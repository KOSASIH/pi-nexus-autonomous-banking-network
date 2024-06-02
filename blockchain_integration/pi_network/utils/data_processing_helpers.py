import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    scaler = StandardScaler()
    data[['feature1', 'feature2', 'feature3']] = scaler.fit_transform(data[['feature1', 'feature2', 'feature3']])
    return data

def split_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def handle_missing_values(data):
    data.fillna(data.mean(), inplace=True)
    return data

def convert_data_type(data, column, data_type):
    data[column] = data[column].astype(data_type)
    return data

def aggregate_data(data, column, aggregation_function):
    return data.groupby(column).agg(aggregation_function)

def merge_data(data1, data2, on):
    return pd.merge(data1, data2, on=on)

def filter_data(data, column, value):
    return data[data[column] == value]

def sort_data(data, column, ascending):
    return data.sort_values(by=column, ascending=ascending)

def group_data(data, column):
    return data.groupby(column)

def pivot_data(data, index, columns, values):
    return data.pivot(index=index, columns=columns, values=values)
