import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def create_model(input_shape, num_classes):
    # Define the machine learning model architecture
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_data, test_data, epochs=10):
    # Train the machine learning model
    model.fit(train_data, epochs=epochs, validation_data=test_data)
    return model

def evaluate_model(model, test_data):
    # Evaluate the machine learning model
    y_pred = model.predict(test_data)
    y_pred_class = tf.argmax(y_pred, axis=1)
    y_true = test_data['label']
    accuracy = accuracy_score(y_true, y_pred_class)
    report = classification_report(y_true, y_pred_class)
    matrix = confusion_matrix(y_true, y_pred_class)
    return accuracy, report, matrix
