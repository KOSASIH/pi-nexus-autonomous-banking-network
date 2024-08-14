import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_squared_log_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import median_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import mean_gamma_deviance
from sklearn.metrics import mean_tweedie_deviance
from sklearn.metrics import mean_pinball_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_squared_log_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import median_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import mean_gamma_deviance
from sklearn.metrics import mean_tweedie_deviance
from sklearn.metrics import mean_pinball_loss

class NodeOptimizer:
    def __init__(self, node_data):
        self.node_data = node_data
        self.model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.tsne = TSNE(n_components=2)
        self.kmeans = KMeans(n_clusters=5)
        self.grid_search = GridSearchCV(estimator=self.model, param_grid={}, cv=5, scoring='neg_mean_squared_error')

    def preprocess_data(self):
        self.node_data = self.scaler.fit_transform(self.node_data)
        self.node_data = self.pca.fit_transform(self.node_data)
        self.node_data = self.tsne.fit_transform(self.node_data)

    def train_model(self):
        X = self.node_data.drop(['latency'], axis=1)
        y = self.node_data['latency']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
        print('R2 Score:', r2_score(y_test, y_pred))

    def tune_hyperparameters(self):
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10]
        }
        self.grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        self.grid_search.fit(self.node_data.drop(['latency'], axis=1), self.node_data['latency'])
        print('Best Parameters:', self.grid_search.best_params_)
        print('Best Score:', self.grid_search.best_score_)

    def evaluate_model(self):
        X = self.node_data.drop(['latency'], axis=1)
        y = self.node_data['latency']
        y_pred = self.model.predict(X)
        print('Mean Squared Error:', mean_squared_error(y, y_pred))
        print('R2 Score:', r2_score(y, y_pred))
        print('Mean Absolute Error:', mean_absolute_error(y, y_pred))
        print('Median Absolute Error:', median_absolute_error(y, y_pred))
        print('Explained Variance Score:', explained_variance_score(y, y_pred))
        print('Max Error:', max_error(y, y_pred))
        print('Mean Squared Log Error:', mean_squared_log_error(y, y_pred))
        print('Median Squared Log Error:', median_squared_log_error(y, y_pred)) 
