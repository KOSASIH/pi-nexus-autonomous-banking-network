import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
dataset = pd.read_csv('credit_data.csv')

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset.drop('credit_score', axis=1), dataset['credit_score'], test_size=0.2, random_state=42)

# Create an LSTM model for credit risk assessment
input_layer = Input(shape=(X_train.shape[1], 1))
x = LSTM(units=50, return_sequences=True)(input_layer)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=input_layer, outputs=x)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_class = tf.round(y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred_class))
print("Classification Report:")
print(classification_report(y_test, y_pred_class))

# Use SHAP values for explainable AI
import shap
explainer = shap.KernelExplainer(model.predict, X_train)
shap_values = explainer.shap_values(X_test)
print("SHAP values:", shap_values)
