import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class AIPoweredInvestmentAdvisor:
    def __init__(self, user_data, market_data):
        self.user_data = user_data
        self.market_data = market_data

    def train_model(self):
        # Load user data and market data into a pandas dataframe
        df = pd.concat([self.user_data, self.market_data], axis=1)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

        # Train a random forest classifier model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        return model

    def make_recommendations(self, model, user_input):
        # Use the trained model to make investment recommendations
        predictions = model.predict(user_input)

        # Return a list of recommended investments
        return predictions

# Example usage
user_data = pd.DataFrame({'risk_tolerance': [3], 'investment_goals': ['long_term']})
market_data = pd.DataFrame({'stock_prices': [100, 200, 300], 'market_trends': ['bullish', 'bearish', 'neutral']})

advisor = AIPoweredInvestmentAdvisor(user_data, market_data)
model = advisor.train_model()
recommendations = advisor.make_recommendations(model, user_data)

print(recommendations)
