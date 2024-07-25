# Importing necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF

# Class for recommendation system
class PiNetworkRecommendationSystem:
    def __init__(self):
        self.model = None

    # Function to train the model
    def train(self, data):
        # Training the model
        self.model = NMF(n_components=10, random_state=42)
        self.model.fit(data)

        # Evaluating the model
        predictions = self.model.predict(data)
        print("Reconstruction Error:", self.model.reconstruction_err_)
        print("Frobenius Norm:", self.model.frobenius_norm_)

    # Function to make recommendations
    def recommend(self, user_id, num_recommendations):
        # Getting the user's preferences
        user_preferences = data[data['user_id'] == user_id]

        # Computing the similarity matrix
        similarity_matrix = cosine_similarity(user_preferences, data)

        # Getting the top N recommendations
        recommendations = similarity_matrix.argsort()[:num_recommendations]
        return recommendations

# Example usage
data = pd.read_csv('recommendation_data.csv')
recommender = PiNetworkRecommendationSystem()
recommender.train(data)
