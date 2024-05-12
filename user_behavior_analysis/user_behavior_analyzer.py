class UserBehaviorAnalyzer:
    def __init__(self, data_preparation, clustering):
        self.data_preparation = data_preparation
        self.clustering = clustering

    def analyze_user_behavior(self, data_file, features, num_clusters):
        """
        Analyzes user behavior using the clustering algorithm.
        """
        data = self.data_preparation.prepare_data(data_file)

        user_clusters = self.clustering.cluster_users(data, features, num_clusters)

        return user_clusters
