import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class DifferentialPrivacy:
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon

    def privatize_data(self, data):
        noisy_data = data + np.random.laplace(0, self.epsilon, size=data.shape)
        return noisy_data

    @staticmethod
    def train_model(X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        svm = SVC()
        svm.fit(X_scaled, y)
        return svm

    @staticmethod
    def evaluate_model(model, X, y):
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return accuracy

# Example usage
if __name__ == "__main__":
    dp = DifferentialPrivacy(epsilon=1.0)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, size=100)
    noisy_X = dp.privatize_data(X)
    model = dp.train_model(noisy_X, y)
    accuracy = dp.evaluate_model(model, noisy_X, y)
    print(f"Model Accuracy: {accuracy:.3f}")
