# Class for sensor data analyzer
class PiNetworkSensorDataAnalyzer:
    def __init__(self):
        self.model = None

    # Function to train the model
    def train(self, data):
        # Training the model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(data.drop('label', axis=1), data['label'])

        # Evaluating the model
        predictions = self.model.predict(data.drop('label', axis=1))
        print("Accuracy:", accuracy_score(data['label'], predictions))
        print("Classification Report:")
        print(classification_report(data['label'], predictions))
        print("Confusion Matrix:")
        print(confusion_matrix(data['label'], predictions))

    # Function to analyze sensor data
    def analyze(self, data):
        # Making predictions
        predictions = self.model.predict(data)
        return predictions

# Example usage
data = pd.read_csv('sensor_data.csv')
analyzer = PiNetworkSensorDataAnalyzer()
analyzer.train(data)
