import unittest
from app.encryption_utils import generate_key, encrypt_file, decrypt_file
import os

class TestEncryptionUtils(unittest.TestCase):
    def setUp(self):
        self.password = 'mysecretpassword'
        self.salt = b'mysalt'
        self.file_path = 'test_file.txt'

    def test_generate_key(self):
        key = generate_key(self.password, self.salt)
        self.assertIsNotNone(key)

    def test_encrypt_file(self):
        with open(self.file_path, 'w') as file:
            file.write('Hello, World!')
        key = generate_key(self.password, self.salt)
        encrypt_file(key, self.file_path)
        with open(self.file_path, 'r') as file:
            encrypted_data = file.read()
            self.assertNotEqual(encrypted_data, 'Hello, World!')

    def test_decrypt_file(self):
        with open(self.file_path, 'w') as file:
            file.write('Hello, World!')
        key = generate_key(self.password, self.salt)
        encrypt_file(key, self.file_path)
        decrypt_file(key, self.file_path)
        with open(self.file_path, 'r') as file:
            decrypted_data = file.read()
            self.assertEqual(decrypted_data, 'Hello, World!')

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

if __name__ == '__main__':
    unittest.main()
