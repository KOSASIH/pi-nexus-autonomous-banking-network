import numpy as np
from sklearn.ensemble import IsolationForest

class IsolationForestAnomalyDetector:
    def __init__(self, n_estimators=100, max_samples=1.0, contamination=0.1, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.iforest = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, random_state=random_state)

    def fit(self, X):
        self.iforest.fit(X)

    def predict(self, X):
        return self.iforest.predict(X)

    def decision_function(self, X):
        return self.iforest.decision_function(X)

    def score_samples(self, X):
        return self.iforest.score_samples(X)
