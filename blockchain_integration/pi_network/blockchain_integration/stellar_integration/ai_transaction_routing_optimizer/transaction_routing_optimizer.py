import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class TransactionRoutingOptimizer:
    def __init__(self, graph, transaction_data):
        self.graph = graph
        self.transaction_data = transaction_data
        self.model = RandomForestRegressor(n_estimators=100)

    def train_model(self):
        X = self.transaction_data.drop(['route_cost'], axis=1)
        y = self.transaction_data['route_cost']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

    def optimize_route(self, source, destination, amount):
        shortest_path = nx.shortest_path(self.graph, source, destination, weight='weight')
        route_cost = self.model.predict([amount, len(shortest_path)])
        return shortest_path, route_cost

    def update_model(self, new_transaction_data):
        self.transaction_data = pd.concat([self.transaction_data, new_transaction_data])
        self.train_model()
