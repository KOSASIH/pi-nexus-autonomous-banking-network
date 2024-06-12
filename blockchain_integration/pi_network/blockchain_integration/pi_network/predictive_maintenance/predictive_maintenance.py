import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class PredictiveMaintenance:
    def __init__(self, iot_data):
        self.iot_data = iot_data
        self.model = RandomForestClassifier()

    def train_model(self):
        X = self.iot_data.drop(['label'], axis=1)
        y = self.iot_data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

    def predict_failure(self, new_data):
        prediction = self.model.predict(new_data)
        return prediction

# Example usage:
predictive_maintenance = PredictiveMaintenance(iot_data)
predictive_maintenance.train_model()
new_data = pd.DataFrame({'temperature': [50], 'vibration': [20]})
failure_prediction = predictive_maintenance.predict_failure(new_data)
print(failure_prediction)
