import unittest
from app.auth_utils import hash_password, verify_password, generate_token, verify_token

class TestAuthUtils(unittest.TestCase):
    def setUp(self):
        self.password = 'mysecretpassword'
        self.user_id = 1
        self.secret_key = 'mysecretkey'

    def test_hash_password(self):
        hashed_password = hash_password(self.password)
        self.assertIsNotNone(hashed_password)

    def test_verify_password(self):
        hashed_password = hash_password(self.password)
        self.assertTrue(verify_password(self.password, hashed_password))

    def test_generate_token(self):
        token = generate_token(self.user_id, self.secret_key)
        self.assertIsNotNone(token)

    def test_verify_token(self):
        token = generate_token(self.user_id, self.secret_key)
        user_id = verify_token(token, self.secret_key)
        self.assertEqual(user_id, self.user_id)

if __name__ == '__main__':
    unittest.main()
