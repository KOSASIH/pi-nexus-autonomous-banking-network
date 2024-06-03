import os
import json
from cryptography.fernet import Fernet

class BackupAndRecovery:
    def __init__(self, wallet_data):
        self.wallet_data = wallet_data
        self.backup_dir = "backups"
        self.recovery_key = Fernet.generate_key()

    def create_backup(self):
        # Create a backup of the wallet data
        backup_data = json.dumps(self.wallet_data)
        encrypted_backup = self.encrypt_backup(backup_data)
        self.save_backup(encrypted_backup)

    def encrypt_backup(self, backup_data):
        # Encrypt the backup data using Fernet
        f = Fernet(self.recovery_key)
        encrypted_backup = f.encrypt(backup_data.encode())
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
        # Decrypt the backup data using Fernet
        f = Fernet(self.recovery_key)
        decrypted_backup = f.decrypt(encrypted_backup).decode()
        return decrypted_backup

if __name__ == '__main__':
    wallet_data = {"accounts": [{"id": 1, "balance": 100}]}
    backup_and_recovery = BackupAndRecovery(wallet_data)
    backup_and_recovery.create_backup()
    backup_and_recovery.recover_backup()
    print("Recovered wallet data:", wallet_data)
