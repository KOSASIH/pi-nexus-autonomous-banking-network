from crypto.key_manager import KeyManager
from models.model import Model
from clients.client import Client

def main():
    key_manager = KeyManager(key_size=2048, ciphertext_size=2048)
    key_manager.generate_keys()

    model = Model(HomomorphicEncryption(key_manager.get_public_key(), key_manager.get_private_key()))

    client = Client(key_manager, model)

    # Generate some sample data
    data = [1, 2, 3, 4, 5]

    # Perform homomorphic encryption-based machine learning
    results = client.perform_homomorphic_ml(data)

    print("Results:", results)

if __name__ == "__main__":
    main()
