from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class MachineLearningModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = RandomForestClassifier(n_estimators=100)

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
