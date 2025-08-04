# isolation_forest.py

import numpy as np
from sklearn.ensemble import IsolationForest


def detect_anomalies(data):
    clf = IsolationForest(random_state=42)
    clf.fit(data)
    scores = clf.decision_function(data)
    labels = clf.predict(data)
    return labels, scores
