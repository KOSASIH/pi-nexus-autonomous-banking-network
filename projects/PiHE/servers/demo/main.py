from crypto.key_manager import KeyManager
from crypto.homomorphic_encryption import HomomorphicEncryption
from models.model import Model
from servers.server import Server
from clients.client import Client

def main():
    key_manager = KeyManager(key_size=2048, ciphertext_size=2048)
    key_manager.generate_keys()

    he = HomomorphicEncryption(key_manager.get_public_key(), key_manager.get_private_key())

    model = Model(he)
    server = Server(model)

    client = Client(key_manager, model)

    # Generate some sample data
    data = [1, 2, 3, 4, 5]

    # Encrypt the data on the client-side
    encrypted_data = client.encrypt_data(data)

    # Send the encrypted data to the server
    encrypted_results = server.perform_homomorphic_ml(encrypted_data)

    # Decrypt the results on the client-side
    decrypted_results = client.decrypt_results(encrypted_results)

    print("Decrypted results:", decrypted_results)

if __name__ == "__main__":
    main()
