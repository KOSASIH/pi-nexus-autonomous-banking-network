import tensorflow as tf
from ai_model_architectures import CNNModel, RNNModel, TransformerModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_ai_model(X, y, model_architecture, epochs=10, batch_size=32):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = model_architecture(input_shape=X.shape[1:], num_classes=len(set(y)))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_scaled, y_test))
    return model, history

def evaluate_ai_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = tf.keras.metrics.Accuracy()
    accuracy.update_state(y_test, y_pred)
    return accuracy.result().numpy()
