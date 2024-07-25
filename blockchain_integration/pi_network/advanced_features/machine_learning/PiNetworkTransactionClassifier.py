# Importing necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Class for transaction classification
class PiNetworkTransactionClassifier:
    def __init__(self):
        self.model = None

    # Function to train the model
    def train(self, data):
        # Preprocessing data
        X = data.drop(['label'], axis=1)
        y = data['label']

        # Scaling data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Splitting data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Training the model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluating the model
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    # Function to make predictions
    def predict(self, data):
        # Preprocessing data
        X = data.drop(['label'], axis=1)

        # Scaling data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Making predictions
        y_pred = self.model.predict(X_scaled)
        return y_pred

    # Function to train a neural network model
    def train_neural_network(self, data):
        # Preprocessing data
        X = data.drop(['label'], axis=1)
        y = data['label']

        # Scaling data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Splitting data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Training the neural network model
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(8, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        # Evaluating the model
        y_pred = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

# Example usage
data = pd.read_csv('transaction_data.csv')
classifier = PiNetworkTransactionClassifier()
classifier.train(data)
