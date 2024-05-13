from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class NexusModel:
    """Nexus AI model definition."""

    def __init__(self, input_shape: tuple, num_classes: int):
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_shape=input_shape))
        self.model.add(Dense(num_classes, activation='softmax'))

    def compile(self) -> None:
        """Compile the model."""
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
