import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

class AccountRecommender:
  def __init__(self, accounts_data):
    self.accounts_data = pd.read_csv(accounts_data)

  def train_model(self):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize)

    # Fit the vectorizer to the account data
    X = vectorizer.fit_transform(self.accounts_data['description'])

    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(X)

    # Store the similarity matrix
    self.similarity_matrix = similarity_matrix

  def recommend_accounts(self, account_id, num_recommendations):
    # Retrieve the account data
    account_data = self.accounts_data.loc[account_id]

    # Compute the similarity scores
    similarity_scores = self.similarity_matrix[account_id]

    # Get the top-N similar accounts
    top_similar_accounts = np.argsort(-similarity_scores)[:num_recommendations]

    return top_similar_accounts

if __name__ == '__main__':
  recommender = AccountRecommender('accounts_data.csv')
  recommender.train_model()
  account_id = 1234567890
  num_recommendations = 5
  recommended_accounts = recommender.recommend_accounts(account_id, num_recommendations)
  print(f'Recommended accounts for account {account_id}: {recommended_accounts}')
