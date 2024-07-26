# dex_project_ai_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class DexProjectAIModel:
    def __init__(self):
        pass

    def train_model(self, data):
        # Train AI model on DEX data
        df = pd.DataFrame(data)
        X = df.drop(['value'], axis=1)
        y = df['value']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'MSE: {mse:.2f}')
        return model

    def make_predictions(self, model, data):
        # Make predictions on new DEX data
        df = pd.DataFrame(data)
        X = df.drop(['value'], axis=1)
        y_pred = model.predict(X)
        return y_pred
