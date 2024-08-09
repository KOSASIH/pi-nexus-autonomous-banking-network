# behavioral_analysis.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from collections import Counter

class BehavioralAnalysis:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.features = ['mouse_movement', 'keyboard_input', 'screen_interaction', 'network_traffic']
        self.target = 'anomaly'

    def preprocess_data(self):
        """
        Preprocess the data by scaling and transforming the features.
        """
        scaler = StandardScaler()
        self.data[self.features] = scaler.fit_transform(self.data[self.features])

        pca = PCA(n_components=0.95)
        self.data[self.features] = pca.fit_transform(self.data[self.features])

        tsne = TSNE(n_components=2, random_state=42)
        self.data[self.features] = tsne.fit_transform(self.data[self.features])

    def train_model(self):
        """
        Train a random forest classifier on the preprocessed data.
        """
        X = self.data[self.features]
        y = self.data[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def evaluate_model(self):
        """
        Evaluate the performance of the trained model.
        """
        y_pred = self.model.predict(self.data[self.features])
        y_true = self.data[self.target]

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        matrix = confusion_matrix(y_true, y_pred)

        print(f'Accuracy: {accuracy:.3f}')
        print(f'Report:\n{report}')
        print(f'Matrix:\n{matrix}')

    def analyze_behavior(self, user_id: int):
        """
        Analyze the behavior of a specific user.
        """
        user_data = self.data[self.data['user_id'] == user_id]

        mouse_movement = user_data['mouse_movement'].values
        keyboard_input = user_data['keyboard_input'].values
        screen_interaction = user_data['screen_interaction'].values
        network_traffic = user_data['network_traffic'].values

        entropy_mouse = entropy(mouse_movement)
        entropy_keyboard = entropy(keyboard_input)
        entropy_screen = entropy(screen_interaction)
        entropy_network = entropy(network_traffic)

        print(f'User {user_id} behavior analysis:')
        print(f'  Mouse movement entropy: {entropy_mouse:.3f}')
        print(f'  Keyboard input entropy: {entropy_keyboard:.3f}')
        print(f'  Screen interaction entropy: {entropy_screen:.3f}')
        print(f'  Network traffic entropy: {entropy_network:.3f}')

        counter = Counter(user_data['anomaly'].values)
        print(f'  Anomaly distribution: {counter}')

    def visualize_behavior(self, user_id: int):
        """
        Visualize the behavior of a specific user.
        """
        user_data = self.data[self.data['user_id'] == user_id]

        plt.figure(figsize=(12, 6))

        sns.scatterplot(x=user_data['mouse_movement'], y=user_data['keyboard_input'], hue=user_data['anomaly'])
        plt.title(f'User {user_id} behavior visualization')
        plt.xlabel('Mouse movement')
        plt.ylabel('Keyboard input')
        plt.show()

if __name__ == '__main__':
    data = pd.read_csv('behavioral_data.csv')
    analysis = BehavioralAnalysis(data)

    analysis.preprocess_data()
    analysis.train_model()
    analysis.evaluate_model()

    user_id = 123
    analysis.analyze_behavior(user_id)
    analysis.visualize_behavior(user_id)
