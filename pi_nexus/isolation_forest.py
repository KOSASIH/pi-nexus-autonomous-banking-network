# isolation_forest.py

from sklearn.ensemble import IsolationForest
import numpy as np

def detect_anomalies(data):
    clf = IsolationForest(random_state=42)
    clf.fit(data)
    scores = clf.decision_function(data)
    labels = clf.predict(data)
    return labels, scores
