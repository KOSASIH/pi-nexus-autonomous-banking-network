# File: apt_detection_dl.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

class APTDetector:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.model = self.build_model()

    def build_model(self):
        # Define deep learning model
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(100,)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        # Train deep learning model
        data = pd.read_csv(self.data_path)
        X = data.drop(['label'], axis=1)
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    def predict(self, data):
        # Predict using deep learning model
        data = self.scaler.transform(data)
        pred = self.model.predict(data)
        return pred

# Example usage:
detector = APTDetector('data.csv', 'odel.h5')
detector.train()
data = pd.read_csv('new_data.csv')
pred = detector.predict(data)
print(pred)
