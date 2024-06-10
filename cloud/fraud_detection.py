import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

class FraudDetection:
    def __init__(self, data):
        self.data = data

    def preprocess(self):
        # Preprocess data using PCA and normalization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.95)
        self.data = pca.fit_transform(self.data)

    def train_model(self):
        # Train a Random Forest Classifier
        X_train, X_test, y_train, y_test = train_test_split(self.data.drop('label', axis=1), self.data['label'], test_size=0.2, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
        print("Random Forest Classification Report:")
        print(classification_report(y_test, y_pred))

        # Train a LSTM Neural Network
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.data.shape[1], 1)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        print("LSTM Accuracy:", model.evaluate(X_test, y_test)[1])

    def predict(self, new_data):
        # Use the trained models to predict fraud probability
        new_data = pca.transform(new_data)
        rf_pred = rf.predict_proba(new_data)[:, 1]
        lstm_pred = model.predict(new_data)
        return rf_pred, lstm_pred

# Example usage
data = pd.read_csv('fraud_data.csv')
fd = FraudDetection(data)
fd.preprocess()
fd.train_model()
new_data = pd.read_csv('new_transactions.csv')
rf_pred, lstm_pred = fd.predict(new_data)
print("Fraud Probability (RF):", rf_pred)
print("Fraud Probability (LSTM):", lstm_pred)
