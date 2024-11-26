import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class NetworkOptimizer:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()

    def optimize(self):
        scaled_data = self.scaler.fit_transform(self.data)
        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(scaled_data)
        return clusters

# Example usage
if __name__ == "__main__":
    data = np.random.rand(100, 5)  # Simulated network performance data
    optimizer = NetworkOptimizer(data)
    optimized_clusters = optimizer.optimize()
    print("Optimized Network Clusters:", optimized_clusters)
