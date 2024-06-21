import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from py2neo import Graph

class AccountProfiler:
  def __init__(self, graph_uri, graph_user, graph_password):
    self.graph = Graph(graph_uri, auth=(graph_user, graph_password))

  def train_model(self, accounts_data):
    # Load the accounts data
    df = pd.read_csv(accounts_data)

    # Preprocess the data
    X = df.drop(['account_id', 'profile'], axis=1)
    y = df['profile']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    accuracy = clf.score(X_test, y_test)
    print(f'Model accuracy: {accuracy:.3f}')

    # Store the model in the graph database
    self.graph.run("CREATE (m:Model {accuracy: {accuracy}})", accuracy=accuracy)
    self.graph.run("CREATE (m)-[:TRAINED_ON {data: {data}}]->(d:Data)", data=accounts_data)

  def predict_profile(self, account_id):
    # Retrieve the account data from the graph database
    account_data = self.graph.run("MATCH (a:Account {id: {account_id}}) RETURN a.data", account_id=account_id).data()

    # Preprocess the account data
    X = pd.DataFrame(account_data, columns=['feature1', 'feature2', ...])

    # Predict the profile using the trained model
    clf = self.graph.run("MATCH (m:Model) RETURN m").data()[0]['m']
    profile = clf.predict(X)

    return profile

if __name__ == '__main__':
  profiler = AccountProfiler('bolt://localhost:7687', 'neo4j', 'password')
  profiler.train_model('accounts_data.csv')
  account_id = '1234567890'
  profile = profiler.predict_profile(account_id)
  print(f'Predicted profile for account {account_id}: {profile}')
