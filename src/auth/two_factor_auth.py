import pyotp
import base64
import os
import json
from datetime import datetime, timedelta

class TwoFactorAuth:
    def __init__(self, user_id):
        """
        Initialize the TwoFactorAuth class for a specific user.

        :param user_id: The unique identifier for the user.
        """
        self.user_id = user_id
        self.secret = self.generate_secret()
        self.totp = pyotp.TOTP(self.secret)

    def generate_secret(self):
        """
        Generate a random base32 secret for the user.

        :return: Base32 encoded secret.
        """
        return base64.b32encode(os.urandom(10)).decode('utf-8')

    def get_qr_code_url(self):
        """
        Generate a URL for the QR code that can be scanned by an authenticator app.

        :return: URL for the QR code.
        """
        return self.totp.provisioning_uri(name=self.user_id, issuer_name="MyApp")

    def generate_token(self):
        """
        Generate a new TOTP token.

        :return: The current TOTP token.
        """
        return self.totp.now()

    def verify_token(self, token):
        """
        Verify a provided TOTP token.

        :param token: The token to verify.
        :return: True if the token is valid, False otherwise.
        """
        return self.totp.verify(token)

    def save_secret(self, filename):
        """
        Save the user's secret to a file.

        :param filename: The name of the file to save the secret.
        """
        with open(filename, 'w') as f:
            json.dump({"user_id": self.user_id, "secret": self.secret}, f)
        print(f"Secret saved to {filename}")

    def load_secret(self, filename):
        """
        Load the user's secret from a file.

        :param filename: The name of the file to load the secret from.
        """
        with open(filename, 'r') as f:
            data = json.load(f)
            self.user_id = data['user_id']
            self.secret = data['secret']
            self.totp = pyotp.TOTP(self.secret)
        print(f"Secret loaded from {filename}")

# Example usage
if __name__ == "__main__":
    user_id = "user@example.com"
    
    # Initialize 2FA for the user
    tfa = TwoFactorAuth(user_id)

    # Generate a QR code URL
    qr_code_url = tfa.get_qr_code_url()
    print("QR Code URL:", qr_code_url)

    # Generate a token
    token = tfa.generate_token()
    print("Generated Token:", token)

    # Verify the token
    is_valid = tfa.verify_token(token)
    print("Is the token valid?", is_valid)

    # Save the secret to a file
    tfa.save_secret("user_secret.json")

    # Load the secret from a file
    tfa.load_secret("user_secret.json")
