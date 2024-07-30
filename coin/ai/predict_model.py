from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_predict_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=10))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
