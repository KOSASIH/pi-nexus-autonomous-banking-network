import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class AIFraudDetection:
    def __init__(self, training_data: pd.DataFrame):
        self.training_data = training_data
        self.rf_model = RandomForestClassifier(n_estimators=100)
        self.nn_model = Sequential([
            Dense(64, activation='relu', input_shape=(training_data.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

    def train(self):
        self.rf_model.fit(self.training_data.drop('target', axis=1), self.training_data['target'])
        self.nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.nn_model.fit(self.training_data.drop('target', axis=1), self.training_data['target'], epochs=10)

    def predict(self, transaction_data: pd.DataFrame) -> list:
        rf_predictions = self.rf_model.predict(transaction_data)
        nn_predictions = self.nn_model.predict(transaction_data)
        combined_predictions = [rf + nn for rf, nn in zip(rf_predictions, nn_predictions)]
        return combined_predictions
