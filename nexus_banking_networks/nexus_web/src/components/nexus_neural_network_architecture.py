import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model


class NexusNeuralNetworkArchitecture:

    def __init__(self):
        self.input_layer = Input(shape=(784,))
        self.hidden_layer1 = Dense(256, activation="relu")(self.input_layer)
        self.dropout1 = Dropout(0.2)(self.hidden_layer1)
        self.hidden_layer2 = Dense(128, activation="relu")(self.dropout1)
        self.dropout2 = Dropout(0.2)(self.hidden_layer2)
        self.output_layer = Dense(10, activation="softmax")(self.dropout2)

    def create_model(self):
        model = Model(inputs=self.input_layer, outputs=self.output_layer)
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model
