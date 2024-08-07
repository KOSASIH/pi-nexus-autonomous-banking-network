import numpy as np
from sklearn.neighbors import LocalOutlierFactor

class LocalOutlierFactorAnomalyDetector:
    def __init__(self, n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski', p=2):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.lof = LocalOutlierFactor(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, metric=metric, p=p)

    def fit(self, X):
        self.lof.fit(X)

    def predict(self, X):
        return self.lof.predict(X)

    def decision_function(self, X):
        return self.lof.decision_function(X)

    def score_samples(self, X):
        return self.lof.score_samples(X)
