import unittest
import pyotp
from src.auth.two_factor_auth import TwoFactorAuth

class TestTwoFactorAuth(unittest.TestCase):
    def setUp(self):
        """Set up a TwoFactorAuth instance for testing."""
        self.user_id = "test_user@example.com"
        self.tfa = TwoFactorAuth(self.user_id)

    def test_generate_secret(self):
        """Test that a secret is generated and is of the correct length."""
        self.assertIsNotNone(self.tfa.secret)
        self.assertEqual(len(self.tfa.secret), 16)  # Base32 secret should be 16 characters

    def test_get_qr_code_url(self):
        """Test that the QR code URL is generated correctly."""
        qr_code_url = self.tfa.get_qr_code_url()
        self.assertIn(self.user_id, qr_code_url)
        self.assertIn("MyApp", qr_code_url)

    def test_generate_token(self):
        """Test that a token is generated and is valid."""
        token = self.tfa.generate_token()
        self.assertIsNotNone(token)
        self.assertEqual(len(token), 6)  # TOTP tokens are typically 6 digits

    def test_verify_token_valid(self):
        """Test that a valid token is verified correctly."""
        token = self.tfa.generate_token()
        self.assertTrue(self.tfa.verify_token(token))

    def test_verify_token_invalid(self):
        """Test that an invalid token is not verified."""
        self.assertFalse(self.tfa.verify_token("123456"))  # Assuming this token is invalid

    def test_load_save_secret(self):
        """Test saving and loading the secret."""
        filename = "test_secret.json"
        self.tfa.save_secret(filename)

        # Create a new instance and load the secret
        new_tfa = TwoFactorAuth(self.user_id)
        new_tfa.load_secret(filename)

        self.assertEqual(self.tfa.secret, new_tfa.secret)

if __name__ == "__main__":
    unittest.main()
