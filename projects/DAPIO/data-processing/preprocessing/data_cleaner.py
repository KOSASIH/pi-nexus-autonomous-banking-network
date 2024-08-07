import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataCleaner:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def handle_missing_values(self) -> pd.DataFrame:
        self.data.fillna(self.data.mean(), inplace=True)
        return self.data

    def remove_outliers(self) -> pd.DataFrame:
        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1
        self.data = self.data[~((self.data < (Q1 - 1.5 * IQR)) | (self.data > (Q3 + 1.5 * IQR)))]
        return self.data

    def scale_data(self) -> pd.DataFrame:
        scaler = StandardScaler()
        self.data[['feature1', 'feature2', 'feature3']] = scaler.fit_transform(self.data[['feature1', 'feature2', 'feature3']])
        return self.data

    def preprocess_data(self) -> pd.DataFrame:
        self.data = self.handle_missing_values()
        self.data = self.remove_outliers()
        self.data = self.scale_data()
        return self.data
