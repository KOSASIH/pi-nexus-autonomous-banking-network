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
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class EonixMachineLearning:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None

    def load_data(self, df):
        self.df = df

    def preprocess_data(self):
        numeric_features = self.df.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.df.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', pd.get_dummies)])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        X = self.df.drop('target', axis=1)
        y = self.df['target']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)

    def feature_selection(self, k):
        selector = SelectKBest(chi2, k=k)
        self.X_train = selector.fit_transform(self.X_train, self.y_train)
        self.X_test = selector.transform(self.X_test)

    def dimensionality_reduction(self, method):
        if method == 'pca':
            reducer = PCA(n_components=2)
        elif method == 'tsne':
            reducer = TSNE(n_components=2)
        elif method == 'lda':
            reducer = LinearDiscriminantAnalysis(n_components=2)

        self.X_train = reducer.fit_transform(self.X_train)
        self.X_test = reducer.transform(self.X_test)

    def train_model(self, model_name):
        if model_name == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100)
        elif model_name == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100)
        elif model_name == 'vm':
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
            param_grid = {'n_estimators': [10, 50, 100, 200], 'ax_depth': [None, 5, 10, 15]}
        elif model_name == 'gradient_boosting':
            param_grid = {'n_estimators': [10, 50, 100, 200], 'learning_rate': [0.1, 0.5, 1]}
        elif model_name == 'vm':
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
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.X_train[:, 0], y=self.X_train[:, 1], hue=self.y_train)
        plt.title("Training Data")
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.X_test[:, 0], y=self.X_test[:, 1], hue=self.y_test)
        plt.title("Testing Data")
        plt.show()

    def save_model(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)

# Example usage:
eonix_ml = EonixMachineLearning()
eonix_ml.load_data(pd.read_csv("data.csv"))
eonix_ml.preprocess_data()
eonix_ml.feature_selection(5)
eonix_ml.dimensionality_reduction("pca")
eonix_ml.train_model("random_forest")
eonix_ml.evaluate_model()
eonix_ml.tune_hyperparameters("random_forest")
eonix_ml.visualize_data()
eonix_ml.save_model("model.pkl")
eonix_ml.load_model("model.pkl")
