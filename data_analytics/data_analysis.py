import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class DataAnalysis:
    def __init__(self, data):
        self.data = data

    def analyze_data(self):
        # Perform data cleaning and preprocessing
        self.data = self.data.dropna()
        self.data = pd.get_dummies(self.data, columns=['transaction_type'])

        # Perform data analysis
        kmeans = KMeans(n_clusters=3, random_state=0).fit(self.data[['amount', 'frequency']])
        self.data['cluster'] = kmeans.labels_

        # Perform statistical analysis
        summary_stats = self.data.describe()
        summary_stats.loc['count'] = len(self.data)
        summary_stats.loc['mean'] = np.mean(self.data)
        summary_stats.loc['std'] = np.std(self.data)

        return summary_stats
