import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class RiskManager:
    def __init__(self, transaction_data):
        self.transaction_data = transaction_data
        self.model = self.train_model()

    def train_model(self):
        # Load and preprocess transaction data
        X = pd.get_dummies(self.transaction_data.drop('risk_level', axis=1))
        y = self.transaction_data['risk_level']

        # Train a Random Forest Classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)

        # Train a Neural Network
        nn_model = Sequential()
        nn_model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
        nn_model.add(Dense(32, activation='relu'))
        nn_model.add(Dense(1, activation='sigmoid'))
        nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        nn_model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        return rf_model, nn_model

    def predict_risk(self, new_transaction):
        # Use the trained models to predict the risk level
        rf_pred = self.model[0].predict(new_transaction)
        nn_pred = self.model[1].predict(new_transaction)
        return (rf_pred + nn_pred) / 2

# Example usage
transaction_data = pd.read_csv('transactions.csv')
risk_manager = RiskManager(transaction_data)
new_transaction = pd.DataFrame({'amount': [100], 'category': ['withdrawal']})
risk_level = risk_manager.predict_risk(new_transaction)
print(f'Predicted risk level: {risk_level:.2f}')
