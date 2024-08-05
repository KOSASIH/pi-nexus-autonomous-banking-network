import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from eonix_ai_models import EonixAIModel, EonixAIModelV2, EonixAIModelV3
from eonix_ai_data_preprocessing import preprocess_data, split_data

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """
    Train the model using the training data and validate on the validation data.

    Args:
        model (keras.Model): The model to train
        X_train (pd.DataFrame): The training data
        y_train (pd.Series): The training labels
        X_val (pd.DataFrame): The validation data
        y_val (pd.Series): The validation labels
        epochs (int): Number of epochs to train
        batch_size (int): Batch size for training

    Returns:
        history (keras.callbacks.History): Training history
    """
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Define the callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, min_delta=0.001),
        keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy')
    ]

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_data=(X_val, y_val), callbacks=callbacks)

    return history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data.

    Args:
        model (keras.Model): The trained model
        X_test (pd.DataFrame): The test data
        y_test (pd.Series): The test labels

    Returns:
        accuracy (float): Accuracy on the test data
        precision (float): Precision on the test data
        recall (float): Recall on the test data
        f1 (float): F1 score on the test data
    """
    predictions = model.predict(X_test)
    predictions_class = tf.argmax(predictions, axis=1)
    accuracy = accuracy_score(y_test, predictions_class)
    precision = precision_score(y_test, predictions_class, average='macro')
    recall = recall_score(y_test, predictions_class, average='macro')
    f1 = f1_score(y_test, predictions_class, average='macro')
    return accuracy, precision, recall, f1

def main():
    # Load the data
    data = pd.read_csv('data.csv')

    # Preprocess the data
    data_preprocessed = preprocess_data(data, categorical_cols=['category'], numerical_cols=['feature1', 'feature2'], text_cols=['text'], target_col='target')

    # Split the data into training, validation, and testing sets
    X_train, X_val, y_train, y_val = split_data(data_preprocessed, target_col='target', test_size=0.2, random_state=42)
    X_test, y_test = split_data(data_preprocessed, target_col='target', test_size=0.1, random_state=42)

    # Create the model
    model = EonixAIModelV2(num_classes=10, input_shape=(X_train.shape[1],))

    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32)

    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
    print(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')

if __name__ == '__main__':
    main()
