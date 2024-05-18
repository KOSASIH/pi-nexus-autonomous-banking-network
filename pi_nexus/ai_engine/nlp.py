# ai_engine/nlp.py
import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Dense, Embedding, SpatialDropout1D


class NLPModel:
    def __init__(self, vocab_size, embedding_dim, max_seq_len):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.model = tf.keras.Sequential(
            [
                Embedding(
                    self.vocab_size, self.embedding_dim, input_length=self.max_seq_len
                ),
                SpatialDropout1D(0.4),
                LSTM(128, dropout=0.2, recurrent_dropout=0.2),
                Dense(10, activation="softmax"),
            ]
        )
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
