# character_animator.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.models import Model

class CharacterAnimator:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        # Input layer
        input_layer = Input(shape=(None, 3))

        # LSTM layer
        lstm_layer = LSTM(128, return_sequences=True)(input_layer)
        lstm_layer = Dropout(0.2)(lstm_layer)

        # Output layer
        output_layer = Dense(3, activation='tanh')(lstm_layer)

        # Model
        model = Model(input_layer, output_layer)
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def animate_character(self, motion_data):
        return self.model.predict(motion_data)
