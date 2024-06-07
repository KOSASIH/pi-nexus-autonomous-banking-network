import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from lime.lime_tabular import LimeTabularExplainer

class ExplainableAnomalyDetector:
    def __init__(self, data, seq_len=100):
        self.data = data
        self.seq_len = seq_len
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100)
        self.explainer = LimeTabularExplainer(self.data, feature_names=['feature_1', 'feature_2',...])

    def train(self):
        X_train, y_train = self.prepare_data()
        self.model.fit(X_train, y_train)

    def prepare_data(self):
        data_scaled = self.scaler.fit_transform(self.data)
        X, y = [], []
        for i in range(len(data_scaled) - self.seq_len):
            X.append(data_scaled[i:i + self.seq_len])
            y.append(0)  # 0 for normal, 1 for anomaly
        X, y = np.array(X), np.array(y)
        return X, y

    def predict(self, data):
        data_scaled = self.scaler.transform(data)
        X_pred = data_scaled[-self.seq_len:]
        X_pred = X_pred.reshape(1, X_pred.shape[0], 1)
        pred = self.model.predict(X_pred)
        return pred[0]

    def explain_anomaly(self, data):
        exp = self.explainer.explain_instance(data, self.model.predict_proba, num_features=5)
        return exp.as_list()

    def detect_anomalies(self, data, threshold=0.5):
        pred = self.predict(data)
        if pred > threshold:
            explanation = self.explain_anomaly(data)
            return True, explanation
        else:
            return False, None
