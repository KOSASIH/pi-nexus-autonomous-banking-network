import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class NodeClustering:
    def __init__(self, node_data):
        self.node_data = node_data
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.kmeans = KMeans(n_clusters=8, random_state=42)

    def preprocess_data(self):
        scaled_data = self.scaler.fit_transform(self.node_data)
        pca_data = self.pca.fit_transform(scaled_data)
        return pca_data

    def cluster_nodes(self):
        pca_data = self.preprocess_data()
        self.kmeans.fit(pca_data)
        node_clusters = self.kmeans.predict(pca_data)
        return node_clusters

    def visualize_clusters(self):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        pca_data = self.preprocess_data()
        tsne = TSNE(n_components=2, random_state=42)
        tsne_data = tsne.fit_transform(pca_data)

        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=self.kmeans.labels_)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Node Clusters")
        plt.show()


node_data = ...  # load node data
node_clustering = NodeClustering(node_data)
node_clusters = node_clustering.cluster_nodes()
node_clustering.visualize_clusters()
