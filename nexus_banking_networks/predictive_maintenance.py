# predictive_maintenance.py
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

class PredictiveMaintenance:
  def __init__(self):
    self.model = keras.Sequential([
      keras.layers.LSTM(50, input_shape=(10, 1)),
      keras.layers.Dense(1)
    ])
    self.model.compile(loss='mean_squared_error', optimizer='adam')

  def train(self, data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    self.model.fit(data_scaled, epochs=100)

  def predict(self, data):
    data_scaled = scaler.transform(data)
    prediction = self.model.predict(data_scaled)
    return prediction

# Example usage:
pm = PredictiveMaintenance()
data = # load ATM sensor data
pm.train(data)

new_data = # new ATM sensor data
prediction = pm.predict(new_data)
print("Predicted maintenance need:", prediction)
