import socket
import pickle
from qeb import quantum_encrypt, quantum_decrypt

# Define QEB client class
class QEBClient:
    def __init__(self, node_id):
        self.node_id = node_id
        self.qeb_node = QEBNode(node_id)

    def send_transaction(self, transaction_data):
        # Encrypt transaction data with quantum encryption
        encrypted_data, quantum_key = quantum_encrypt(transaction_data)

        # Send encrypted data to QEB node
        self.qeb_node.add_transaction(encrypted_data)

    def retrieve_transaction(self, transaction_data):
        # Retrieve encrypted data from QEB node
        encrypted_data = self.qeb_node.retrieve_transaction(transaction_data)

        # Decrypt encrypted data with quantum decryption
        decrypted_data = quantum_decrypt(encrypted_data, quantum_key)

        # Return decrypted data
        return decrypted_data

# Create QEB client instance
qeb_client = QEBClient("client1")

# Send transaction to QEB node
qeb_client.send_transaction("Transaction data 1")

# Retrieve transaction from QEB node
decrypted_data = qeb_client.retrieve_transaction("Transaction data 1")

print(decrypted_data)
