import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class EonixAI:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_data(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def preprocess_data(self):
        numeric_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.X_train.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', pd.get_dummies)])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        self.X_train = preprocessor.fit_transform(self.X_train)
        self.X_test = preprocessor.transform(self.X_test)

    def train_model(self, model_type):
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(kernel='rbf', C=1, gamma=0.1)
        elif model_type == 'naive_bayes':
            self.model = GaussianNB()
        elif model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=42)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        elif model_type == 'neural_network':
            self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        else:
            raise ValueError("Invalid model type")

        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))

    def visualize_data(self):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_train)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=self.y_train)
        plt.title("PCA Visualization")
        plt.show()

        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(self.X_train)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=self.y_train)
        plt.title("t-SNE Visualization")
        plt.show()
