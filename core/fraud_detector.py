import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

class FraudDetector:
  def __init__(self, data):
    self.data = data
    self.scaler = StandardScaler()
    self.model = self.create_model()

  def create_model(self):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(self.data.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

  def train_model(self):
    X = self.data.drop(['is_fraud'], axis=1)
    y = self.data['is_fraud']
    X_scaled = self.scaler.fit_transform(X)
    self.model.fit(X_scaled, y, epochs=10, batch_size=32, validation_split=0.2)

  def predict_fraud(self, transaction):
    X = pd.DataFrame([transaction], columns=self.data.columns[:-1])
    X_scaled = self.scaler.transform(X)
    prediction = self.model.predict(X_scaled)
    return prediction[0]

# Load the data
data = pd.read_csv('fraud_data.csv')

# Create the fraud detector
fraud_detector = FraudDetector(data)

# Train the model
fraud_detector.train_model()

# Use the trained model to predict fraud
transaction = {'amount': 1000, 'country': 'USA', 'card_type': 'VISA'}
prediction = fraud_detector.predict_fraud(transaction)
print(f'Fraud prediction: {prediction}')
