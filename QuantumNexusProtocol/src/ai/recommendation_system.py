import pandas as pd
from sklearn.neighbors import NearestNeighbors

class RecommendationSystem:
    def __init__(self, data):
        self.data = data

    def recommend(self, user_id, n_recommendations=5):
        model = NearestNeighbors(metric='cosine')
        model.fit(self.data)
        distances, indices = model.kneighbors(self.data.loc[user_id].values.reshape(1, -1), n_neighbors=n_recommendations + 1)
        return self.data.index[indices.flatten()[1:]].tolist()

# Example usage
if __name__ == "__main__":
    data = pd.DataFrame(np.random.rand(10, 5))  # Simulated user-item interaction data
    recommender = RecommendationSystem(data)
    recommendations = recommender.recommend(user_id=0)
    print(f"Recommendations for User 0: {recommendations}")
