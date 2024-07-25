# Importing necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Class for artificial intelligence
class PiNetworkArtificialIntelligence:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    # Function to train the model
    def train(self, data):
        X = data.drop('target', axis=1)
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        return self.model.score(X_test, y_test)

    # Function to make predictions
    def predict(self, data):
        return self.model.predict(data)

# Example usage
ai = PiNetworkArtificialIntelligence()
data = pd.read_csv('data.csv')
accuracy = ai.train(data)
print(f"Accuracy: {accuracy:.4f}")
