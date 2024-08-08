from crypto.key_manager import KeyManager
from crypto.homomorphic_encryption import HomomorphicEncryption
from models.model import Model

class Client:
    def __init__(self, key_manager: KeyManager, model: Model):
        self.key_manager = key_manager
        self.model = model
        self.he = HomomorphicEncryption(key_manager.get_public_key(), key_manager.get_private_key())

    def encrypt_data(self, data):
        encrypted_data = []
        for value in data:
            encrypted_value = self.he.encrypt(value)
            encrypted_data.append(encrypted_value)
        return encrypted_data

    def send_data_to_model(self, encrypted_data):
        # Send the encrypted data to the model for training or prediction
        return self.model.train(encrypted_data) if len(encrypted_data) > 1 else self.model.predict(encrypted_data)

    def decrypt_results(self, encrypted_results):
        decrypted_results = self.model.decrypt_predictions(encrypted_results)
        return decrypted_results

    def perform_homomorphic_ml(self, data):
        encrypted_data = self.encrypt_data(data)
        encrypted_results = self.send_data_to_model(encrypted_data)
        decrypted_results = self.decrypt_results(encrypted_results)
        return decrypted_results
