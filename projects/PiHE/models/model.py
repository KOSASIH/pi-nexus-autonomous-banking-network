from crypto.homomorphic_encryption import HomomorphicEncryption

class Model:
    def __init__(self, he: HomomorphicEncryption):
        self.he = he
        self.weights = []  # encrypted weights
        self.bias = 0  # encrypted bias

    def train(self, X, y):
        # Train the model using the encrypted data
        # For simplicity, let's assume a linear model
        encrypted_weights = self.he.encrypt(1)  # initialize weights to 1
        encrypted_bias = self.he.encrypt(0)  # initialize bias to 0

        for x, y in zip(X, y):
            encrypted_x = self.he.encrypt(x)
            encrypted_y = self.he.encrypt(y)

            # Compute the encrypted prediction
            encrypted_prediction = self.he.add(self.he.multiply(encrypted_x, encrypted_weights), encrypted_bias)

            # Compute the encrypted error
            encrypted_error = self.he.subtract(encrypted_y, encrypted_prediction)

            # Update the encrypted weights and bias
            encrypted_weights = self.he.add(encrypted_weights, self.he.multiply(encrypted_x, encrypted_error))
            encrypted_bias = self.he.add(encrypted_bias, encrypted_error)

        self.weights = encrypted_weights
        self.bias = encrypted_bias

    def predict(self, X):
        # Make predictions using the encrypted model
        encrypted_predictions = []
        for x in X:
            encrypted_x = self.he.encrypt(x)
            encrypted_prediction = self.he.add(self.he.multiply(encrypted_x, self.weights), self.bias)
            encrypted_predictions.append(encrypted_prediction)
        return encrypted_predictions

    def decrypt_predictions(self, encrypted_predictions):
        # Decrypt the predictions using the homomorphic encryption scheme
        decrypted_predictions = []
        for encrypted_prediction in encrypted_predictions:
            decrypted_prediction = self.he.decrypt(encrypted_prediction)
            decrypted_predictions.append(decrypted_prediction)
        return decrypted_predictions
