import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.sparse import csr_matrix

class NodeRanking:
    def __init__(self, nodes, transactions):
        self.nodes = nodes
        self.transactions = transactions
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.kmeans = KMeans(n_clusters=8)

    def calculate_node_features(self):
        node_texts = [node['description'] for node in self.nodes]
        node_vectors = self.vectorizer.fit_transform(node_texts)
        node_vectors = self.scaler.fit_transform(node_vectors.toarray())
        node_vectors = self.pca.fit_transform(node_vectors)
        return node_vectors

    def calculate_transaction_features(self):
        transaction_texts = [transaction['metadata']['description'] for transaction in self.transactions]
        transaction_vectors = self.vectorizer.transform(transaction_texts)
        transaction_vectors = self.scaler.transform(transaction_vectors.toarray())
        transaction_vectors = self.pca.transform(transaction_vectors)
        return transaction_vectors

    def calculate_similarity_matrix(self, node_vectors, transaction_vectors):
        similarity_matrix = cosine_similarity(node_vectors, transaction_vectors)
        return similarity_matrix

    def calculate_node_ranks(self, similarity_matrix):
        node_ranks = np.sum(similarity_matrix, axis=1)
        return node_ranks

    def cluster_nodes(self, node_vectors):
        kmeans_labels = self.kmeans.fit_predict(node_vectors)
        return kmeans_labels

    def calculate_silhouette_score(self, node_vectors, kmeans_labels):
        silhouette = silhouette_score(node_vectors, kmeans_labels)
        return silhouette

    def run(self):
        node_vectors = self.calculate_node_features()
        transaction_vectors = self.calculate_transaction_features()
        similarity_matrix = self.calculate_similarity_matrix(node_vectors, transaction_vectors)
        node_ranks = self.calculate_node_ranks(similarity_matrix)
        kmeans_labels = self.cluster_nodes(node_vectors)
        silhouette = self.calculate_silhouette_score(node_vectors, kmeans_labels)
        return node_ranks, kmeans_labels, silhouette

if __name__ == '__main__':
    nodes = pd.read_csv('nodes.csv')
    transactions = pd.read_csv('transactions.csv')
    node_ranking = NodeRanking(nodes, transactions)
    node_ranks, kmeans_labels, silhouette = node_ranking.run()
    print('Node Ranks:', node_ranks)
    print('KMeans Labels:', kmeans_labels)
    print('Silhouette Score:', silhouette)
