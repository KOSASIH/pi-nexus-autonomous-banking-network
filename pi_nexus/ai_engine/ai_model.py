# ai_engine/ai_model.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten

class AIModel:
    def __init__(self):
        self.model = tf.keras.Sequential([
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(10, 1)),
            MaxPooling1D(pool_size=2),
            LSTM(64, activation='tanh'),
            Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
