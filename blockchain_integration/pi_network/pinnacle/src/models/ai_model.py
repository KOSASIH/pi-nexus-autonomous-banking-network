import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class AIModel:
    def __init__(self, model_type: str, features: List[str], target: str):
        self.model_type = model_type
        self.features = features
        self.target = target
        self.model = None

    def train(self, data: pd.DataFrame) -> None:
        X = data[self.features]
        y = data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
        else:
            raise ValueError("Invalid model type")

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(data[self.features])

    def evaluate(self, data: pd.DataFrame) -> float:
        if self.model is None:
            raise ValueError("Model not trained")
        y_pred = self.model.predict(data[self.features])
        return self.model.score(data[self.features], data[self.target])
