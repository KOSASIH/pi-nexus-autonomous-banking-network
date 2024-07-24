import numpy as np
import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        data = pd.read_csv(self.file_path)
        X = data.drop(['target'], axis=1)
        y = data['target']
        return X, y

    def preprocess_data(self, X, y):
        X = X / 255.0
        y = tf.keras.utils.to_categorical(y, num_classes=10)
        return X, y

    def split_data(self, X, y, test_size=0.2):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
