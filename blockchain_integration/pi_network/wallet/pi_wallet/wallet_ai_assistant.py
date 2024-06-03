import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class WalletAIAssistant:
    def __init__(self, user_data):
        self.user_data = user_data

    def train_model(self):
        # Extract features from user data
        features = self.user_data[
            ["income", "expenses", "credit_score", "transaction_history"]
        ]
        labels = self.user_data["financial_goal"]

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Train random forest classifier
        model = RandomForestClassifier(n_estimators=100)
        model.fit(features_scaled, labels)

        return model

    def provide_advice(self, user_input):
        # Extract features from user input
        user_features = np.array(
            [
                user_input["income"],
                user_input["expenses"],
                user_input["credit_score"],
                user_input["transaction_history"],
            ]
        )

        # Scale features
        scaler = StandardScaler()
        user_features_scaled = scaler.transform(user_features.reshape(1, -1))

        # Make prediction
        model = self.train_model()
        prediction = model.predict(user_features_scaled)

        # Provide personalized financial advice
        if prediction == 0:
            return "Based on your financial profile, we recommend increasing your emergency fund to 3-6 months' worth of expenses."
        elif prediction == 1:
            return "Based on your financial profile, we recommend investing in a diversified portfolio to achieve your long-term goals."
        else:
            return "Based on your financial profile, we recommend reducing your debt by consolidating high-interest loans and credit cards."


if __name__ == "__main__":
    user_data = pd.read_csv("user_data.csv")
    wallet_ai_assistant = WalletAIAssistant(user_data)

    user_input = {
        "income": 50000,
        "expenses": 30000,
        "credit_score": 750,
        "transaction_history": ["rent", "utilities", "groceries"],
    }

    advice = wallet_ai_assistant.provide_advice(user_input)
    print(advice)
