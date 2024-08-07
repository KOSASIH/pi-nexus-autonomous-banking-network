import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.ai_model import AIModel

class AIUtils:
    def __init__(self, ai_model: AIModel):
        self.ai_model = ai_model

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        data[['feature1', 'feature2', 'feature3']] = scaler.fit_transform(data[['feature1', 'feature2', 'feature3']])
        return data

    def split_data(self, data: pd.DataFrame, target: str, test_size: float = 0.2) -> tuple:
        X = data.drop(target, axis=1)
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def evaluate_model(self, data: pd.DataFrame, target: str) -> float:
        X = data.drop(target, axis=1)
        y = data[target]
        return self.ai_model.evaluate(X, y)
