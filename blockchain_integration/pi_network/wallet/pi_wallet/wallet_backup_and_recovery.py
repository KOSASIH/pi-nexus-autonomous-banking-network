import base64
import json
import os
from hashlib import sha256

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class BackupAndRecovery:
    def __init__(self, wallet_data, private_key_path, public_key_path):
        self.wallet_data = wallet_data
        self.private_key_path = private_key_path
        self.public_key_path = public_key_path
        self.backup_dir = "backups"
        self.recovery_key = self.generate_recovery_key()

    def generate_recovery_key(self):
        # Generate a recovery key using RSA
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        with open(self.private_key_path, "wb") as f:
            f.write(private_pem)

        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH,
        )
        with open(self.public_key_path, "wb") as f:
            f.write(public_pem)

        return private_key

    def create_backup(self):
        # Create a backup of the wallet data
        backup_data = json.dumps(self.wallet_data)
        encrypted_backup = self.encrypt_backup(backup_data)
        self.save_backup(encrypted_backup)

    def encrypt_backup(self, backup_data):
        # Encrypt the backup data using RSA-OAEP
        encrypted_backup = self.recovery_key.encrypt(
            backup_data.encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return encrypted_backup

    def save_backup(self, encrypted_backup):
        # Save the encrypted backup to a file
        backup_file = os.path.join(self.backup_dir, "wallet_backup.json")
        with open(backup_file, "wb") as f:
            f.write(encrypted_backup)

    def recover_backup(self):
        # Recover the wallet data from a backup file
        backup_file = os.path.join(self.backup_dir, "wallet_backup.json")
        with open(backup_file, "rb") as f:
            encrypted_backup = f.read()
        decrypted_backup = self.decrypt_backup(encrypted_backup)
        self.wallet_data = json.loads(decrypted_backup)

    def decrypt_backup(self, encrypted_backup):
        # Decrypt the backup data using RSA-OAEP
        decrypted_backup = self.recovery_key.decrypt(
            encrypted_backup,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return decrypted_backup.decode()


if __name__ == "__main__":
    wallet_data = {"accounts": [{"id": 1, "balance": 100}]}
    private_key_path = "private_key.pem"
    public_key_path = "public_key.pem"
    backup_and_recovery = BackupAndRecovery(
        wallet_data, private_key_path, public_key_path
    )
    backup_and_recovery.create_backup()
    backup_and_recovery.recover_backup()
    print("Recovered wallet data:", wallet_data)
