import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class StellarTransactionPredictor:
    def __init__(self, horizon_url, network_passphrase):
        self.horizon_url = horizon_url
        self.network_passphrase = network_passphrase
        self.client = stellar_sdk.Client(horizon_url, network_passphrase)

    def fetch_transactions(self, start_time, end_time):
        transactions = self.client.transactions(start_time, end_time)
        df = pd.DataFrame([tx.to_dict() for tx in transactions])
        return df

    def train_model(self, df):
        X = df.drop(['hash', 'result'], axis=1)
        y = df['result']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def predict_transaction(self, model, transaction_data):
        prediction = model.predict(transaction_data)
        return prediction
