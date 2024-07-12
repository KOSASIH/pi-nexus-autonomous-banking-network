import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class ArtificialIntelligence:

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))

    def predict(self, X):
        return self.model.predict(X)


ai = ArtificialIntelligence()
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)
ai.train(X, y)
y_pred = ai.predict(X)
print(y_pred)
