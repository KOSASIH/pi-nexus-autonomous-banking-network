from crypto.key_manager import KeyManager
from crypto.homomorphic_encryption import HomomorphicEncryption
from models.model import Model

def main():
    key_manager = KeyManager(key_size=2048, ciphertext_size=2048)
    key_manager.generate_keys()

    he = HomomorphicEncryption(key_manager.get_public_key(), key_manager.get_private_key())

    model = Model(he)

    # Generate some sample data
    X = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    # Train the model
    model.train(X, y)

    # Make predictions
    encrypted_predictions = model.predict(X)

    # Decrypt the predictions
    decrypted_predictions = model.decrypt_predictions(encrypted_predictions)

    print("Decrypted predictions:", decrypted_predictions)

if __name__ == "__main__":
    main()
