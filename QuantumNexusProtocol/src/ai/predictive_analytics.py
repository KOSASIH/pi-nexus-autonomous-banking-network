import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class PredictiveAnalytics:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestRegressor()

    def train_model(self):
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        return self.model.score(X_test, y_test)

# Example usage
if __name__ == "__main__":
    data = pd.DataFrame(np.random.rand(100, 6), columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'target'])
    analytics = PredictiveAnalytics(data)
    accuracy = analytics.train_model()
    print("Model Accuracy:", accuracy)
