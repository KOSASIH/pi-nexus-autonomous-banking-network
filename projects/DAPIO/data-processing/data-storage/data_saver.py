import pandas as pd
from sklearn.externals import joblib

class DataSaver:
    def __init__(self, data: pd.DataFrame, model: RandomForestClassifier):
        self.data = data
        self.model = model

        def save_data(self, file_path: str) -> None:
        self.data.to_csv(file_path, index=False)

    def save_model(self, file_path: str) -> None:
        joblib.dump(self.model, file_path)
