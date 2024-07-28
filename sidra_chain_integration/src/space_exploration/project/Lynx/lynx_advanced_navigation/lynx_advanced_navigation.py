import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class AdvancedNavigation:
    def __init__(self, navigation_data):
        self.navigation_data = navigation_data

    def train_model(self):
        X = self.navigation_data.drop(['target'], axis=1)
        y = self.navigation_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'MSE: {mse:.2f}')
        return model

    def predict(self, model, new_data):
        new_data = pd.DataFrame(new_data)
        prediction = model.predict(new_data)
        return prediction

# Example usage:
navigation_data = pd.read_csv('navigation_data.csv')
advanced_navigation = AdvancedNavigation(navigation_data)
model = advanced_navigation.train_model()
new_data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
prediction = advanced_navigation.predict(model, new_data)
print(prediction)
