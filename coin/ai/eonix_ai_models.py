import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class EonixAIModel(keras.Model):
    def __init__(self, num_classes, input_shape):
        super(EonixAIModel, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)
        self.max_pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.max_pool2 = layers.MaxPooling2D((2, 2))
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu')
        self.max_pool3 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.2)
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.max_pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        return self.dense2(x)

class EonixAIModelV2(keras.Model):
    def __init__(self, num_classes, input_shape):
        super(EonixAIModelV2, self).__init__()
        self.lstm1 = layers.LSTM(128, return_sequences=True, input_shape=input_shape)
        self.lstm2 = layers.LSTM(64)
        self.dense1 = layers.Dense(64, activation='relu')
        self.dropout1 = layers.Dropout(0.2)
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        return self.dense2(x)

class EonixAIModelV3(keras.Model):
    def __init__(self, num_classes, input_shape):
        super(EonixAIModelV3, self).__init__()
        self.transformer = layers.TransformerEncoderLayer(d_model=128, num_heads=8, dropout=0.1)
        self.dense1 = layers.Dense(64, activation='relu')
        self.dropout1 = layers.Dropout(0.2)
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.transformer(inputs)
        x = self.dense1(x)
        x = self.dropout1(x)
        return self.dense2(x)

def evaluate_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    predictions_class = tf.argmax(predictions, axis=1)
    accuracy = accuracy_score(test_labels, predictions_class)
    precision = precision_score(test_labels, predictions_class, average='macro')
    recall = recall_score(test_labels, predictions_class, average='macro')
    f1 = f1_score(test_labels, predictions_class, average='macro')
    return accuracy, precision, recall, f1
