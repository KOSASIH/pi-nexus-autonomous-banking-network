import hashlib
import secrets
from collections import defaultdict
from cryptography.fernet import Fernet

class HDWallet:
    def __init__(self, seed):
        self.seed = seed
        self.addresses = {}  # index -> (private_key, public_key)

    def generate_address(self, index):
        private_key = secrets.token_hex(32)  # 64 hex characters
        public_key = self._generate_public_key(private_key)
        self.addresses[index] = (private_key, public_key)
        return public_key

    @staticmethod
    def _generate_public_key(private_key):
        return hashlib.sha256(private_key.encode()).hexdigest()

class Wallet:
    def __init__(self):
        self.wallets = {}  # user -> HDWallet
        self.balances = defaultdict(int)  # user -> balance
        self.transaction_history = defaultdict(list)  # user -> list of transactions
        self.key_storage = {}  # user -> encrypted private keys

    def create_wallet(self, user, seed):
        if user in self.wallets:
            raise ValueError("Wallet already exists for this user.")
        
        hd_wallet = HDWallet(seed)
        hd_wallet.generate_address(0)  # Generate the first address
        self.wallets[user] = hd_wallet
        print(f"Wallet created for {user}. Public Key: {hd_wallet.addresses[0][1]}")

    def encrypt_private_key(self, user, private_key):
        key = Fernet.generate_key()
        cipher_suite = Fernet(key)
        encrypted_key = cipher_suite.encrypt(private_key.encode())
        self.key_storage[user] = (encrypted_key, key)
        return encrypted_key

    def decrypt_private_key(self, user):
        encrypted_key, key = self.key_storage[user]
        cipher_suite = Fernet(key)
        return cipher_suite.decrypt(encrypted_key).decode()

    def get_balance(self, user):
        return self.balances[user]

    def deposit(self, user, amount):
        if user not in self.wallets:
            raise ValueError("Wallet does not exist for this user.")
        self.balances[user] += amount
        self.transaction_history[user].append(f"Deposited {amount}. New balance: {self.balances[user]}")
        print(f"Deposited {amount} to {user}'s wallet. New balance: {self.balances[user]}")

    def withdraw(self, user, amount):
        if user not in self.wallets:
            raise ValueError("Wallet does not exist for this user.")
        if self.balances[user] < amount:
            raise ValueError("Insufficient balance.")
        self.balances[user] -= amount
        self.transaction_history[user].append(f"Withdrew {amount}. New balance: {self.balances[user]}")
        print(f"Withdrew {amount} from {user}'s wallet. New balance: {self.balances[user]}")

    def get_wallet_info(self, user):
        if user not in self.wallets:
            raise ValueError("Wallet does not exist for this user.")
        hd_wallet = self.wallets[user]
        return {
            "addresses": hd_wallet.addresses,
            "balance": self.get_balance(user),
            "transaction_history": self.transaction_history[user]
        }

# Example usage
if __name__ == "__main__":
    wallet_manager = Wallet()
    
    # Create wallets for users
    wallet_manager.create_wallet("user1", "seed_phrase_1")
    wallet_manager.create_wallet("user2", "seed_phrase_2")

    # Deposit and withdraw funds
    wallet_manager.deposit("user1", 100)
    wallet_manager.withdraw("user1", 50)

    # Get wallet info
    print(wallet_manager .get_wallet_info("user1"))
