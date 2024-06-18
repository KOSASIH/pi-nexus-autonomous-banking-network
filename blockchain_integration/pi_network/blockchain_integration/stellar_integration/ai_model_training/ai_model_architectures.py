import tensorflow as tf

class CNNModel(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)
        self.max_pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        return self.dense2(x)

class RNNModel(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(RNNModel, self).__init__()
        self.lstm1 = tf.keras.layers.LSTM(128, input_shape=input_shape)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense1 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.lstm1(x)
        x = self.dropout1(x)
        return self.dense1(x)

class TransformerModel(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(TransformerModel, self).__init__()
        self.encoder = tf.keras.layers.TransformerEncoder(input_shape=input_shape, num_heads=8, num_layers=6)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense1 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.encoder(x)
        x = self.dropout1(x)
        return self.dense1(x)
