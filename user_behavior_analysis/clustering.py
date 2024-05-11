import numpy as np
from sklearn.cluster import KMeans

class Clustering:
    def __init__(self, data, features):
        self.data = data
        self.features = features

    def cluster_users(self, num_clusters):
        """
        Clusters users based on their behavior.
        """
        X = self.data[self.features]
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)

        return kmeans.labels_
