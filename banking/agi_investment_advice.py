import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

class AGIInvestmentAdvice:
    def __init__(self):
        self.model= self.create_model()

    def create_model(self):
        input_layer = Input(shape=(10,))
        x = Dense(64,activation='relu')(input_layer)
        x = Dense(64, activation='relu')(x)
        output_layer = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_model(self, data):
        self.model.fit(data, epochs=100, batch_size=32)

    def predict_investment_advice(self, data):
        prediction = self.model.predict(data)
        return prediction

# Example usage:
agi_investment_advice = AGIInvestmentAdvice()
data = np.random.rand(100, 10)
agi_investment_advice.train_model(data)
prediction = agi_investment_advice.predict_investment_advice(data)
print(prediction)
