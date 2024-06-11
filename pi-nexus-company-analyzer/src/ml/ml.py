# ml.py

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

class ML:
    def __init__(self, model_name, data_file):
        self.model_name = model_name
        self.data_file = data_file
        self.data = pd.read_csv(data_file)
        self.X = self.data.drop(['target'], axis=1)
        self.y = self.data['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def train_model(self):
        if self.model_name == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_name == 'vm':
            self.model = SVC(kernel='rbf', C=1, gamma='auto', random_state=42)
        elif self.model_name == 'lp':
            self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        elif self.model_name == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_name == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=42)
        elif self.model_name == 'naive_bayes':
            self.model = GaussianNB()
        elif self.model_name == 'qda':
            self.model = QuadraticDiscriminantAnalysis()
        elif self.model_name == 'gaussian_process':
            self.model = GaussianProcessClassifier(kernel=RBF())
        self.model.fit(self.X_train_scaled, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        matrix = confusion_matrix(self.y_test, y_pred)
        return accuracy, report, matrix

    def save_model(self, model_file):
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, model_file):
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

ml = ML('random_forest', 'data.csv')
ml.train_model()
accuracy, report, matrix = ml.evaluate_model()
print(f'Accuracy: {accuracy:.3f}')
print(f'Report:\n{report}')
print(f'Matrix:\n{matrix}')
