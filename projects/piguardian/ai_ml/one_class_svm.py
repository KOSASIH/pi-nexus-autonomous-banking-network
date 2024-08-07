import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

class OneClassSVMAnomalyDetector:
    def __init__(self, kernel='rbf', gamma=0.1, nu=0.1):
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.svm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.svm.fit(X_scaled)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.svm.predict(X_scaled)

    def decision_function(self, X):
        X_scaled = self.scaler.transform(X)
        return self.svm.decision_function(X_scaled)

    def score_samples(self, X):
        X_scaled = self.scaler.transform(X)
        return self.svm.score_samples(X_scaled)
