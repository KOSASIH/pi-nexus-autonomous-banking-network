from models.model import Model

class Server:
    def __init__(self, model: Model):
        self.model = model

    def receive_data(self, encrypted_data):
        # Receive the encrypted data from the client
        return encrypted_data

    def process_data(self, encrypted_data):
        # Process the encrypted data using the model
        if len(encrypted_data) > 1:
            self.model.train(encrypted_data)
        else:
            return self.model.predict(encrypted_data)

    def send_results(self, encrypted_results):
        # Send the encrypted results back to the client
        return encrypted_results

    def perform_homomorphic_ml(self, encrypted_data):
        encrypted_results = self.process_data(encrypted_data)
        return self.send_results(encrypted_results)
