# File: threat_detection_ml_gnn.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from spektral.data import BatchLoader
from spektral.layers import GCNConv, GlobalAttentionPool

class ThreatDetector:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.rf_model = RandomForestClassifier(n_estimators=100)
        self.gnn_model = self.build_gnn_model()

    def build_gnn_model(self):
        # Define graph neural network model
        x_in = Input(shape=(100, 100))  # Input shape: (num_nodes, num_features)
        x = GCNConv(64, activation='relu')([x_in, x_in])
        x = Dropout(0.2)(x)
        x = GCNConv(32, activation='relu')([x, x_in])
        x = Dropout(0.2)(x)
        x = GlobalAttentionPool(64)(x)
        x = Dense(2, activation='softmax')(x)
        model = Model(inputs=x_in, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        # Train random forest model
        data = pd.read_csv(self.data_path)
        X = data.drop(['label'], axis=1)
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.rf_model.fit(X_train, y_train)

        # Train graph neural network model
        batch_loader = BatchLoader(self.data_path, batch_size=32, mask=True)
        self.gnn_model.fit(batch_loader.load(), epochs=10)

    def predict(self, data):
        # Predict using random forest model
        rf_pred = self.rf_model.predict(data)

        # Predict using graph neural network model
        gnn_pred = self.gnn_model.predict(data)

        # Combine predictions using ensemble method
        pred = np.argmax(rf_pred + gnn_pred, axis=1)
        return pred

# Example usage:
detector = ThreatDetector('data.csv', 'model.h5')
detector.train()
data = pd.read_csv('new_data.csv')
pred = detector.predict(data)
print(pred)
