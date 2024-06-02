import unittest
from pi_network.pi_network_db import PiNetworkDB
from pi_network.utils.cryptographic_helpers import encrypt_data, decrypt_data

class TestPiNetworkDB(unittest.TestCase):
    def setUp(self):
        self.pi_network_db = PiNetworkDB()

    def test_create_transaction(self):
        transaction_data = {'amount': 10.99, 'ender': 'Alice', 'eceiver': 'Bob'}
        encrypted_data = encrypt_data(transaction_data)
        transaction_id = self.pi_network_db.create_transaction(encrypted_data)
        self.assertIsNotNone(transaction_id)

    def test_get_transaction(self):
        transaction_id = 1
        encrypted_data = self.pi_network_db.get_transaction(transaction_id)
        decrypted_data = decrypt_data(encrypted_data)
        self.assertEqual(decrypted_data, {'amount': 10.99, 'ender': 'Alice', 'eceiver': 'Bob'})

    def test_create_user(self):
        user_data = {'username': 'Alice', 'password': 'password123'}
        encrypted_data = encrypt_data(user_data)
        user_id = self.pi_network_db.create_user(encrypted_data)
        self.assertIsNotNone(user_id)

    def test_get_user(self):
        user_id = 1
        encrypted_data = self.pi_network_db.get_user(user_id)
        decrypted_data = decrypt_data(encrypted_data)
        self.assertEqual(decrypted_data, {'username': 'Alice', 'password': 'password123'})

    def test_update_transaction(self):
        transaction_id = 1
        updated_data = {'amount': 20.99, 'ender': 'Alice', 'eceiver': 'Bob'}
        encrypted_data = encrypt_data(updated_data)
        self.pi_network_db.update_transaction(transaction_id, encrypted_data)

    def test_delete_transaction(self):
        transaction_id = 1
        self.pi_network_db.delete_transaction(transaction_id)

if __name__ == '__main__':
    unittest.main()
