import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class DataAnalyzer:
    def __init__(self, data):
        self.data = pd.DataFrame(data)

    def perform_pca(self, n_components):
        pca = PCA(n_components=n_components)
        self.data_pca = pca.fit_transform(self.data)
        return self.data_pca

    def perform_kmeans(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters)
        self.data_kmeans = kmeans.fit_predict(self.data_pca)
        return self.data_kmeans

    def visualize_clusters(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.data_pca[:, 0], self.data_pca[:, 1], c=self.data_kmeans)
        plt.show()
