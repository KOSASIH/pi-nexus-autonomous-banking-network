# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10,)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load transaction data
transaction_data = pd.read_csv('transactions.csv')

# Preprocess data
X = transaction_data.drop(['is_fraud'], axis=1)
y = transaction_data['is_fraud']

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Use the model to predict fraud
def predict_fraud(transaction):
    input_data = pd.DataFrame([transaction], columns=X.columns)
    prediction = model.predict(input_data)
    return prediction[0][0] > 0.5
