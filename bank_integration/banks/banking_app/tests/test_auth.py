import unittest
from flask_jwt_extended import create_access_token
from app.auth import authenticate

class TestAuth(unittest.TestCase):
    def test_authenticate_valid_credentials(self):
        # Test authenticating with valid credentials
        username = "test_user"
        password = "test_password"
        access_token = authenticate(username, password)
        self.assertIsNotNone(access_token)

    def test_authenticate_invalid_credentials(self):
        # Test authenticating with invalid credentials
        username = "invalid_user"
        password = "invalid_password"
        access_token = authenticate(username, password)
        self.assertIsNone(access_token)

    def test_create_access_token(self):
        # Test creating an access token
        username = "test_user"
        access_token = create_access_token(identity=username)
        self.assertIsNotNone(access_token)

if __name__ == "__main__":
    unittest.main()
