import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_X_y
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from joblib import Parallel, delayed
from functools import partial
from collections import defaultdict
from itertools import product
from warnings import warn
from time import time
from datetime import datetime

class Optimization(BaseEstimator):
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.search_space = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10]
        }
        self.parallel_backend = 'multiprocessing'
        self.n_jobs = -1
        self.verbose = 1

    def _objective_function(self, params):
        self.model.set_params(**params)
        self.model.fit(self.X_train_, self.y_train_)
        y_pred = self.model.predict(self.X_val_)
        score = mean_squared_error(self.y_val_, y_pred, squared=False)
        return score

    def _grid_search(self, params):
        grid_search = GridSearchCV(self.model, params, cv=5, scoring='neg_mean_squared_error', n_jobs=self.n_jobs)
        grid_search.fit(self.X_train_, self.y_train_)
        return grid_search.best_params_, grid_search.best_score_

    def _differential_evolution(self, bounds):
        res = differential_evolution(self._objective_function, bounds, popsize=50, tol=1e-5, disp=True)
        return res.x, res.fun

    def fit(self, X, y):
        check_X_y(X, y)
        self.X_train_, self.X_val_, self.y_train_, self.y_val_ = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train_ = self.scaler.fit_transform(self.X_train_)
        self.X_val_ = self.scaler.transform(self.X_val_)

        if self.config['method'] == 'grid_search':
            params = self.search_space
            best_params, best_score = self._grid_search(params)
            self.model.set_params(**best_params)
            self.model.fit(self.X_train_, self.y_train_)
            self.best_score_ = best_score
            self.best_params_ = best_params

        elif self.config['method'] == 'differential_evolution':
            bounds = [(10, 200), (None, 15), (2, 10), (1, 10)]
            best_params, best_score = self._differential_evolution(bounds)
            self.model.set_params(**dict(zip(self.search_space.keys(), best_params)))
            self.model.fit(self.X_train_, self.y_train_)
            self.best_score_ = best_score
            self.best_params_ = best_params

        else:
            raise ValueError("Invalid optimization method. Choose from 'grid_search' or 'differential_evolution'.")

    def predict(self, X):
        check_is_fitted(self, 'best_params_')
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def score(self, X, y):
        check_is_fitted(self, 'best_params_')
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred, squared=False)

def train_test_split(*arrays, **options):
    return tuple(array.split(test_size=options['test_size'], random_state=options['random_state']) for array in arrays)

def check_is_fitted(estimator, attributes):
    if not hasattr(estimator, attributes):
        raise NotFittedError(f"This {estimator.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

if __name__ == '__main__':
    config = {
        'method': 'differential_evolution'
    }
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    optimization = Optimization(config)
    start_time = time()
    optimization.fit(X, y)
    end_time = time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds.")
    print(f"Best score: {optimization.best_score_:.4f}")
    print(f"Best parameters: {optimization.best_params_}")
