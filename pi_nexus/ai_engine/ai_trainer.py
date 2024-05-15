# ai_engine/ai_trainer.py
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class AITrainer:
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config

    def train(self):
        # Train AI model
        X, y = self.data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        early_stopping = EarlyStopping(monitor="val_loss", patience=5)
        model_checkpoint = ModelCheckpoint(self.config.model_file, save_best_only=True)
        self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            callbacks=[early_stopping, model_checkpoint],
        )
