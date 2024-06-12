import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class AICybersecurity:
    def __init__(self, network_data):
        self.network_data = network_data
        self.model = RandomForestClassifier()

    def train_model(self):
        X = self.network_data.drop(['label'], axis=1)
        y = self.network_data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

    def predict_threat(self, new_data):
        prediction = self.model.predict(new_data)
        return prediction

# Example usage:
ai_cybersecurity = AICybersecurity(network_data)
ai_cybersecurity.train_model()
new_data = pd.DataFrame({'packet_size': [100], 'protocol': ['TCP']})
threat_prediction = ai_cybersecurity.predict_threat(new_data)
print(threat_prediction)
