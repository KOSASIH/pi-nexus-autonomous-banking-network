import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM

class EnergyProductionAnalysis:
    def __init__(self, data):
        self.data = data
        self.X = data.drop(['energy_production'], axis=1)
        self.y = data['energy_production']

    def feature_engineering(self):
        # Extract date and time features
        self.X['date'] = pd.to_datetime(self.X['date'])
        self.X['hour'] = self.X['date'].dt.hour
        self.X['day'] = self.X['date'].dt.day
        self.X['month'] = self.X['date'].dt.month
        self.X['year'] = self.X['date'].dt.year

        # Extract weather features
        self.X['temperature'] = self.X['weather_data'].apply(lambda x: x['temperature'])
        self.X['humidity'] = self.X['weather_data'].apply(lambda x: x['humidity'])
        self.X['wind_speed'] = self.X['weather_data'].apply(lambda x: x['wind_speed'])

        # Extract energy production features
        self.X['energy_production_lag1'] = self.y.shift(1)
        self.X['energy_production_lag2'] = self.y.shift(2)
        self.X['energy_production_lag3'] = self.y.shift(3)

        # Scale features using StandardScaler
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

    def dimensionality_reduction(self):
        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=0.95)
        self.X_pca = pca.fit_transform(self.X_scaled)

        # Apply t-SNE to reduce dimensionality
        tsne = TSNE(n_components=2, random_state=42)
        self.X_tsne = tsne.fit_transform(self.X_pca)

    def clustering(self):
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        self.X_kmeans = kmeans.fit_predict(self.X_tsne)

        # Apply One-Class SVM for anomaly detection
        ocsvm = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
        self.X_ocsvm = ocsvm.fit_predict(self.X_tsne)

    def regression(self):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X_pca, self.y, test_size=0.2, random_state=42)

        # Train Random Forest Regressor
        rfr = RandomForestRegressor(n_estimators=100, random_state=42)
        rfr.fit(X_train, y_train)
        y_pred_rfr = rfr.predict(X_test)

        # Train XGBoost Regressor
        xgb = XGBRegressor(n_estimators=100, random_state=42)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)

        # Train CatBoost Regressor
        cb = CatBoostRegressor(n_estimators=100, random_state=42)
        cb.fit(X_train, y_train)
        y_pred_cb = cb.predict(X_test)

        # Train LightGBM Regressor
        lgbm = LGBMRegressor(n_estimators=100, random_state=42)
        lgbm.fit(X_train, y_train)
        y_pred_lgbm = lgbm.predict(X_test)

        # Evaluate models using mean squared error
        mse_rfr = mean_squared_error(y_test, y_pred_rfr)
        mse_xgb = mean_squared_error(y_test, y_pred_xgb)
        mse_cb = mean_squared_error(y_test, y_pred_cb)
        mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)

        print(f'MSE (Random Forest): {mse_rfr:.2f}')
        print(f'MSE (XGBoost): {mse_xgb:.2f}')
        print(f'MSE (CatBoost): {mse_cb:.2f}')
        print(f'MSE (LightGBM): {mse_lgbm:.2f}')

    def visualize(self):
        # Visualize energy production data
        plt.plot(self.y)
        plt.xlabel('Time')
        plt.ylabel('Energy Production')
               plt.title('Energy Production Over Time')
        plt.show()

        # Visualize clustering results
        plt.scatter(self.X_tsne[:, 0], self.X_tsne[:, 1], c=self.X_kmeans)
        plt.xlabel('t-SNE Feature 1')
        plt.ylabel('t-SNE Feature 2')
        plt.title('K-Means Clustering Results')
        plt.show()

        # Visualize anomaly detection results
        plt.scatter(self.X_tsne[:, 0], self.X_tsne[:, 1], c=self.X_ocsvm)
        plt.xlabel('t-SNE Feature 1')
        plt.ylabel('t-SNE Feature 2')
        plt.title('One-Class SVM Anomaly Detection Results')
        plt.show()

    def run(self):
        self.feature_engineering()
        self.dimensionality_reduction()
        self.clustering()
        self.regression()
        self.visualize()

if __name__ == '__main__':
    # Load energy production data
    data = pd.read_csv('energy_production_data.csv')

    # Create EnergyProductionAnalysis object
    epa = EnergyProductionAnalysis(data)

    # Run analysis
    epa.run()
