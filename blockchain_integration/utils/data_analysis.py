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

    def perform_logistic_regression(self, target):
        from sklearn.linear_model import LogisticRegression
        X = self.data.drop(target, axis=1)
        y = self.data[target]
        logistic_regression = LogisticRegression()
        logistic_regression.fit(X, y)
        return logistic_regression

    def perform_decision_tree(self, target):
        from sklearn.tree import DecisionTreeClassifier
        X = self.data.drop(target, axis=1)
        y = self.data[target]
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X, y)
        return decision_tree

    def perform_random_forest(self, target, n_estimators=100):
        from sklearn.ensemble import RandomForestClassifier
        X = self.data.drop(target, axis=1)
        y = self.data[target]
        random_forest = RandomForestClassifier(n_estimators=n_estimators)
        random_forest.fit(X, y)
        return random_forest
