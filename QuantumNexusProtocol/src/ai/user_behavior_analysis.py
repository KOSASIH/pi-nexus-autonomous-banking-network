import pandas as pd
from sklearn.decomposition import PCA

class UserBehaviorAnalyzer:
    def __init__(self, data):
        self.data = data

    def analyze(self):
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.data)
        return reduced_data

# Example usage
if __name__ == "__main__":
    data = pd.read_csv('user_data.csv')  # Load user behavior data
    analyzer = UserBehaviorAnalyzer(data)
    analysis_result = analyzer.analyze()
    print("User  Behavior Analysis Result:", analysis_result)
