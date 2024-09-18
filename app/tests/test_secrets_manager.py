import unittest
from app.secrets_manager import SecretsManager
import os

class TestSecretsManager(unittest.TestCase):
    def setUp(self):
        self.secrets_file = 'secrets.json'
        self.secrets_manager = SecretsManager(self.secrets_file)

    def test_get_secret(self):
        self.secrets_manager.set_secret('mysecret', 'myvalue')
        self.assertEqual(self.secrets_manager.get_secret('mysecret'), 'myvalue')

    def test_set_secret(self):
        self.secrets_manager.set_secret('mysecret', 'myvalue')
        self.assertEqual(self.secrets_manager.get_secret('mysecret'), 'myvalue')

    def tearDown(self):
        if os.path.exists(self.secrets_file):
            os.remove(self.secrets_file)

if __name__ == '__main__':
    unittest.main()
