import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, explained_variance_score, max_error, mean_squared_log_error, median_squared_log_error, mean_absolute_percentage_error, median_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, tSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance, mean_pinball_loss
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import type_of_target
from sklearn.exceptions import NotFittedError
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import iplot
import os
import joblib
import pickle
import gc

class NodeOptimizer(BaseEstimator, RegressorMixin):
    def __init__(self, node_data, target_column, categorical_columns, numerical_columns, text_columns, 
                 model_type='random_forest', hyperparameter_tuning=False, feature_selection=False, 
                 clustering=False, visualization=False, save_model=False, load_model=False):
        self.node_data = node_data
        self.target_column = target_column
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.text_columns = text_columns
        self.model_type = model_type
        self.hyperparameter_tuning = hyperparameter_tuning
        self.feature_selection = feature_selection
        self.clustering = clustering
        self.visualization = visualization
        self.save_model = save_model
        self.load_model = load_model
        self.model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.tsne = TSNE(n_components=2)
        self.kmeans = KMeans(n_clusters=5)
        self.grid_search = GridSearchCV(estimator=self.model, param_grid={}, cv=5, scoring='neg_mean_squared_error')

    def preprocess_data(self):
        self.node_data = self.node_data.dropna()
        self.node_data = self.node_data.drop_duplicates()
        self.node_data = self.node_data.reset_index(drop=True)
        self.node_data[self.categorical_columns] = self.node_data[self.categorical_columns].astype('category')
        self.node_data[self.numerical_columns] = self.node_data[self.numerical_columns].astype('float64')
        self.node_data[self.text_columns] = self.node_data[self.text_columns].astype('str')
        self.node_data = pd.get_dummies(self.node_data, columns=self.categorical_columns)
        self.node_data = self.scaler.fit_transform(self.node_data)
        self.node_data = self.pca.fit_transform(self.node_data)
        self.node_data = self.tsne.fit_transform(self.node_data)

    def train_model(self):
        X = self.node_data.drop([self.target_column], axis=1)
        y = self.node_data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'ada_boost':
            self.model = AdaBoostRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'bagging':
            self.model = BaggingRegressor(n_estimators=100, random_state=42)
        elif
