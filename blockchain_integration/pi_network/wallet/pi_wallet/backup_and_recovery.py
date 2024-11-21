import json
import os

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.serialization import load_pem_private_key


class BackupAndRecovery:
    def __init__(self, wallet_data, private_key_path, public_key_path, encryption_key):
        self.wallet_data = wallet_data
        self.private_key_path = private_key_path
        self.public_key_path = public_key_path
        self.encryption_key = encryption_key

    def create_backup(self):
        # Encrypt the wallet data using a derived key
        fernet = Fernet(self.encryption_key)
        encrypted_data = fernet.encrypt(json.dumps(self.wallet_data).encode())

        # Write the encrypted wallet data to a backup file
        backup_file_path = os.path.join(
            self.wallet_data["backup_dir"], "wallet_backup.json"
        )
        with open(backup_file_path, "wb") as backup_file:
            backup_file.write(encrypted_data)

    def decrypt_backup(self, backup_file_path):
        # Load the private key for decryption
        with open(self.private_key_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(), password=None, backend=default_backend()
            )

        # Decrypt the backup data using the private key
        with open(backup_file_path, "rb") as backup_file:
            encrypted_data = backup_file.read()
        decrypted_data = private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        # Load the decrypted data into a wallet data dictionary
        decrypted_wallet_data = json.loads(decrypted_data.decode())
        return decrypted_wallet_data


if __name__ == "__main__":
    # Create a new wallet data dictionary
    wallet_data = {"wallet_version": "1.0", "backup_dir": "backup_dir"}

    # Generate a new RSA key pair for the wallet
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
    public_key = private_key.public_key()

    # Write the private key to a file
    with open("private_key.pem", "wb") as key_file:
        key_file.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Write the public key to a file
    with open("public_key.pem", "wb") as key_file:
        key_file.write(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )

    # Create a derived key for encrypting the wallet data using Fernet
    encryption_key = Fernet.generate_key()

    # Create a backup of the wallet data
    backup_and_recovery = BackupAndRecovery(
        wallet_data, "private_key.pem", "public_key.pem", encryption_key
    )
    backup_and_recovery.create_backup()

    # Decrypt the backup data
    decrypted_wallet_data = backup_and_recovery.decrypt_backup("wallet_backup.json")

    # Verify that the decrypted wallet data matches the original wallet data
    assert wallet_data == decrypted_wallet_data
