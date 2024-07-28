import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

class ArtificialIntelligence:
    def __init__(self, ai_data):
        self.ai_data = ai_data

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(self.ai_data.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_model(self, model):
        model.fit(self.ai_data, epochs=10, batch_size=32, validation_split=0.2)
        return model

    def predict(self, model, new_data):
        new_data = tf.convert_to_tensor(new_data)
        prediction = model.predict(new_data)
        return prediction

# Example usage:
ai_data = pd.read_csv('ai_data.csv')
artificial_intelligence = ArtificialIntelligence(ai_data)
model = artificial_intelligence.build_model()
model = artificial_intelligence.train_model(model)
new_data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
prediction = artificial_intelligence.predict(model, new_data)
print(prediction)
