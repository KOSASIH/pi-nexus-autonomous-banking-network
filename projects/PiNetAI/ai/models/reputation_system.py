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
        for model_name, prediction in predictions.items():
            accuracy = accuracy_score(self.data['rating'], prediction)
            precision = precision_score(self.data['rating'], prediction, average='macro')
            recall = recall_score(self.data['rating'], prediction, average='macro')
            f1 = f1_score(self.data['rating'], prediction, average='macro')
            metrics[model_name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
        return metrics

    def visualize_results(self, predictions):
        plt.figure(figsize=(10, 6))
        sns.set_style('whitegrid')
        sns.barplot(x=list(predictions.keys()), y=[metrics['accuracy'] for metrics in predictions.values()])
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Model Performance')
        plt.show()

    def optimize_hyperparameters(self):
        def objective(trial):
            params = {
                'Random Forest': {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10)
                },
                'XGBoost': {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e2),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000)
                },
                'CatBoost': {
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e2),
                    'depth': trial.suggest_int('depth', 3, 10)
                },
                'LightGBM': {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e2),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000)
                },
                'Neural Network': {
                    'hidden_layers': trial.suggest_int('hidden_layers', 1, 5),
                    'hidden_units': trial.suggest_int('hidden_units', 10, 100),
                    'dropout_rate': trial.suggest_uniform('dropout_rate', 0.1, 0.5)
                }
            }
            model_name = trial.suggest_categorical('model_name', list(self.models.keys()))
            model = self.models[model_name]
            model.set_params(**params[model_name])
            model.fit(self.data)
            prediction = model.predict(self.data)
            accuracy = accuracy_score(self.data['rating'], prediction)
            return -accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        best_trial = study.best_trial
        best_model_name = best_trial.params['model_name']
        best_model = self.models[best_model_name]
        best_model.set_params(**best_trial.params)
        return best_model

    def calculate_text_features(self, text_data):
        vectorizer = TfidfVectorizer(max_features=1000)
        text_features = vectorizer.fit_transform(text_data)
        return text_features

    def calculate_collaborative_features(self, user_item_matrix):
        collaborative_features = np.zeros((self.num_users, self.num_items))
        for i in range(self.num_users):
            for j in range(self.num_items):
                if user_item_matrix[i, j] > 0:
                    collaborative_features[i, j] = self.calculate_collaborative_feature(i, j, user_item_matrix)
        return collaborative_features

    def calculate_collaborative_feature(self, user_id, item_id, user_item_matrix):
        user_ratings = user_item_matrix[user_id, :]
        item_ratings = user_item_matrix[:, item_id]
        collaborative_feature = np.dot(user_ratings, item_ratings) / (np.linalg.norm(user_ratings) * np.linalg.norm(item_ratings))
        return collaborative_feature

if __name__ == '__main__':
    # Load data
    data = pd.read_csv('data.csv')

    # Create ReputationSystem object
    reputation_system = ReputationSystem(data, num_users=1000, num_items=1000)

    # Preprocess data
    reputation_system.preprocess_data()

    # Create user-item matrices
    user_item_matrix = reputation_system.create_user_item_matrices()

    # Calculate reputation scores
    reputation_scores = reputation_system.calculate_reputation_scores(user_item_matrix)

    # Train models
    reputation_system.train_models(reputation_scores)

    # Predict reputation scores
    predictions = reputation_system.predict_reputation_scores(reputation_scores)

    # Evaluate models
    metrics = reputation_system.evaluate_models(predictions)

    # Visualize results
    reputation_system.visualize_results(metrics)

    # Optimize hyperparameters
    best_model = reputation_system.optimize_hyperparameters()

    # Calculate text features
    text_data = pd.read_csv('text_data.csv')
    text_features = reputation_system.calculate_text_features(text_data)

    # Calculate collaborative features
    collaborative_features = reputation_system.calculate_collaborative_features(user_item_matrix)

    # Combine features
    combined_features = np.concatenate((reputation_scores, text_features, collaborative_features), axis=1)

    # Train final model
    final_model = reputation_system.models['Neural Network']
    final_model.fit(combined_features, reputation_system.data['rating'])

    # Evaluate final model
    final_prediction = final_model.predict(combined_features)
    final_metrics = reputation_system.evaluate_models({'Final Model': final_prediction})

    # Visualize final results
    reputation_system.visualize_results(final_metrics)

    # Save final model
    final_model.save('final_model.h5')
