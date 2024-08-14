import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer

class ReputationSystem:
    def __init__(self, data, num_users, num_items, embedding_dim=128):
        self.data = data
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.user_embeddings = np.random.rand(num_users, embedding_dim)
        self.item_embeddings = np.random.rand(num_items, embedding_dim)
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(objective='binary:logistic', max_depth=6, learning_rate=0.1, n_estimators=1000),
            'CatBoost': CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6),
            'LightGBM': LGBMClassifier(objective='binary', max_depth=6, learning_rate=0.1, n_estimators=1000),
            'Neural Network': self.create_neural_network()
        }

    def create_neural_network(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(self.embedding_dim*2,)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def preprocess_data(self):
        self.data['user_id'] = self.data['user_id'].astype(str)
        self.data['item_id'] = self.data['item_id'].astype(str)
        self.data['rating'] = self.data['rating'].astype(float)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data.sort_values(by='timestamp', inplace=True)

    def create_user_item_matrices(self):
        user_item_matrix = np.zeros((self.num_users, self.num_items))
        for i, row in self.data.iterrows():
            user_item_matrix[row['user_id'], row['item_id']] = row['rating']
        return user_item_matrix

    def calculate_reputation_scores(self, user_item_matrix):
        reputation_scores = np.zeros((self.num_users, self.num_items))
        for i in range(self.num_users):
            for j in range(self.num_items):
                if user_item_matrix[i, j] > 0:
                    reputation_scores[i, j] = self.calculate_reputation_score(i, j, user_item_matrix)
        return reputation_scores

    def calculate_reputation_score(self, user_id, item_id, user_item_matrix):
        user_ratings = user_item_matrix[user_id, :]
        item_ratings = user_item_matrix[:, item_id]
        user_entropy = entropy(user_ratings)
        item_entropy = entropy(item_ratings)
        reputation_score = user_entropy + item_entropy
        return reputation_score

    def train_models(self, reputation_scores):
        X_train, X_test, y_train, y_test = train_test_split(reputation_scores, self.data['rating'], test_size=0.2, random_state=42)
        for model_name, model in self.models.items():
            if model_name == 'Neural Network':
                y_train = to_categorical(y_train)
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
                model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_train), callbacks=[early_stopping])
            else:
                model.fit(X_train, y_train)

    def predict_reputation_scores(self, reputation_scores):
        predictions = {}
        for model_name, model in self.models.items():
            if model_name == 'Neural Network':
                predictions[model_name] = model.predict(reputation_scores)[:, 0]
            else:
                predictions[model_name] = model.predict(reputation_scores)
        return predictions

    def evaluate_models(self, predictions):
        metrics = {}
        for model_name, prediction in
