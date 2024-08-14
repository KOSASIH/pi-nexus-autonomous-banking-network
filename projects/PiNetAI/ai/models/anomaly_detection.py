import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import optuna

class AnomalyDetector:
    def __init__(self, data, contamination=0.1):
        self.data = data
        self.contamination = contamination
        self.models = {
            'Isolation Forest': IsolationForest(contamination=self.contamination),
            'One-Class SVM': OneClassSVM(kernel='rbf', gamma=0.1, nu=self.contamination),
            'Local Outlier Factor': LocalOutlierFactor(n_neighbors=20, contamination=self.contamination),
            'XGBoost': XGBClassifier(objective='binary:logistic', max_depth=6, learning_rate=0.1, n_estimators=1000),
            'CatBoost': CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6),
            'LightGBM': LGBMClassifier(objective='binary', max_depth=6, learning_rate=0.1, n_estimators=1000),
            'Neural Network': self.create_neural_network()
        }

    def create_neural_network(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(self.data.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def preprocess_data(self):
        scaler = StandardScaler()
        self.data[['feature1', 'feature2', 'feature3']] = scaler.fit_transform(self.data[['feature1', 'feature2', 'feature3']])
        pca = PCA(n_components=2)
        self.data[['pca1', 'pca2']] = pca.fit_transform(self.data[['feature1', 'feature2', 'feature3']])
        tsne = TSNE(n_components=2)
        self.data[['tsne1', 'tsne2']] = tsne.fit_transform(self.data[['feature1', 'feature2', 'feature3']])

    def train_models(self):
        for model_name, model in self.models.items():
            if model_name == 'Neural Network':
                X_train, X_test, y_train, y_test = train_test_split(self.data.drop('label', axis=1), self.data['label'], test_size=0.2, random_state=42)
                y_train = to_categorical(y_train)
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
                model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_train), callbacks=[early_stopping])
            else:
                model.fit(self.data)

    def predict_anomalies(self):
        predictions = {}
        for model_name, model in self.models.items():
            if model_name == 'Neural Network':
                predictions[model_name] = model.predict(self.data.drop('label', axis=1))[:, 0]
            else:
                predictions[model_name] = model.predict(self.data)
        return predictions

    def evaluate_models(self, predictions):
        metrics = {}
        for model_name, prediction in predictions.items():
            metrics[model_name] = {
                'accuracy': accuracy_score(self.data['label'], prediction),
                'precision': precision_score(self.data['label'], prediction),
                'recall': recall_score(self.data['label'], prediction),
                'f1': f1_score(self.data['label'], prediction)
            }
        return metrics

    def visualize_results(self, predictions):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for i, (model_name, prediction) in enumerate(predictions.items()):
            sns.scatterplot(x='pca1', y='pca2', hue=prediction, data=self.data, ax=axs[i])
            axs[i].set_title(model_name)
        plt.show()

    def optimize_hyperparameters(self):
        def objective(tr
