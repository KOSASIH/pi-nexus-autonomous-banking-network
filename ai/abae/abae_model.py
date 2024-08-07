import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ABAEModel:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, data):
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

    def predict(self, data):
        return self.model.predict(data)

    def evaluate(self, data):
        X, y = data
        y_pred = self.model.predict(X)
        return accuracy_score(y, y_pred)

# Load dataset
df = pd.read_csv(" dataset.csv")

# Split dataset into features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train model
model = ABAEModel()
model.train((X, y))

# Evaluate model
accuracy = model.evaluate((X, y))
print(f"Model accuracy: {accuracy:.3f}")
