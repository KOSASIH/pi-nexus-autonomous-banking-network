import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EonixMachineLearning:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, df):
        self.df = df

    def preprocess_data(self):
        self.df = pd.get_dummies(self.df)
        X = self.df.drop('target', axis=1)
        y = self.df['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def train_model(self, model_name):
        if model_name == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100)
        elif model_name == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100)
        elif model_name == 'svm':
            self.model = SVC()
        elif model_name == 'logistic_regression':
            self.model = LogisticRegression()
        elif model_name == 'decision_tree':
            self.model = DecisionTreeClassifier()
        elif model_name == 'knn':
            self.model = KNeighborsClassifier()
        elif model_name == 'naive_bayes':
            self.model = GaussianNB()
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        print("ROC-AUC Score:", roc_auc_score(self.y_test, y_pred))

    def tune_hyperparameters(self, model_name):
        if model_name == 'random_forest':
            param_grid = {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 5, 10, 15]}
        elif model_name == 'gradient_boosting':
            param_grid = {'n_estimators': [10, 50, 100, 200], 'learning_rate': [0.1, 0.5, 1]}
        elif model_name == 'svm':
            param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}
        elif model_name == 'logistic_regression':
            param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
        elif model_name == 'decision_tree':
            param_grid = {'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10]}
        elif model_name == 'knn':
            param_grid = {'n_neighbors': [3, 5, 10, 15], 'weights': ['uniform', 'distance']}
        elif model_name == 'naive_bayes':
            param_grid = {'var_smoothing': [1e-9, 1e-6, 1e-3]}
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        print("Best Parameters:", grid_search.best_params_)
        print("Best Score:", grid_search.best_score_)

    def visualize_data(self):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_train)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y_train)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Visualization')
        plt.show()

        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(self.X_train)
        plt.scatter(X_tsne[:, 0],
