# File: ai_cybersecurity_analytics.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

class AICybersecurityAnalytics:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.rf_model = RandomForestClassifier(n_estimators=100)
        self.keras_model = self.build_keras_model()

    def build_keras_model(self):
        # Define Keras model
        input_layer = Input(shape=(100,))
        x = Dense(64, activation='relu')(input_layer)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        output_layer = Dense(2, activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        # Train AI models
        data = pd.read_csv(self.data_path)
        X = data.drop(['label'], axis=1)
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.rf_model.fit(X_train, y_train)
        self.keras_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    def predict(self, data):
        # Predict using AI models
        rf_pred = self.rf_model.predict(data)
        keras_pred = self.keras_model.predict(data)
        return rf_pred, keras_pred

# Example usage:
analytics = AICybersecurityAnalytics('data.csv', 'odel.h5')
analytics.train()
data = pd.read_csv('new_data.csv')
rf_pred, keras_pred = analytics.predict(data)
print(rf_pred, keras_pred)
