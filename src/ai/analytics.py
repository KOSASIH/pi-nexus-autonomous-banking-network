import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

class Analytics:
    def __init__(self, data):
        self.data = data

    def visualize_data(self):
        """Visualize the data using various plots."""
        plt.figure(figsize=(10, 6))
        sns.countplot(x='risk_label', data=self.data)
        plt.title('Distribution of Risk Labels')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.heatmap(self.data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()

    def trend_analysis(self, time_column, value_column):
        """Analyze trends over time."""
        self.data[time_column] = pd.to_datetime(self.data[time_column])
        trend_data = self.data.groupby(time_column)[value_column].sum().reset_index()

        plt.figure(figsize=(12, 6))
        plt.plot(trend_data[time_column], trend_data[value_column], marker='o')
        plt.title('Trend Analysis')
        plt.xlabel('Time')
        plt.ylabel(value_column)
        plt.grid()
        plt.show()

    def detect_anomalies(self, feature_columns):
        """Detect anomalies in the data using Isolation Forest."""
        isolation_forest = IsolationForest(contamination=0.1)
        self.data['anomaly'] = isolation_forest.fit_predict(self.data[feature_columns])
        anomalies = self.data[self.data['anomaly'] == -1]

        plt.figure(figsize=(10, 6))
        plt.scatter(self.data[feature_columns[0]], self.data[feature_columns[1]], color='blue', label='Normal')
        plt.scatter(anomalies[feature_columns[0]], anomalies[feature_columns[1]], color='red', label='Anomaly')
        plt.title('Anomaly Detection')
        plt.xlabel(feature_columns[0])
        plt.ylabel(feature_columns[1])
        plt.legend()
        plt.show()

    def generate_report(self):
        """Generate a summaryreport of the analytics performed."""
        summary = {
            'total_records': len(self.data),
            'risk_distribution': self.data['risk_label'].value_counts().to_dict(),
            'correlation_matrix': self.data.corr().to_dict()
        }
        return summary
