import ntru
from fhe import FHE

class QuantumEncryption:
    def __init__(self, ntru_params: ntru.Params, fhe_params: FHE.Params):
        self.ntru_params = ntru_params
        self.fhe_params = fhe_params

    def encrypt_account_data(self, account_data: bytes) -> bytes:
        # Encrypt account data using NTRU
        encrypted_data = ntru.encrypt(account_data, self.ntru_params)
        # Encrypt encrypted data using FHE
        doubly_encrypted_data = FHE.encrypt(encrypted_data, self.fhe_params)
        return doubly_encrypted_data

    def decrypt_account_data(self, doubly_encrypted_data: bytes) -> bytes:
        # Decrypt doubly encrypted data using FHE
        encrypted_data = FHE.decrypt(doubly_encrypted_data, self.fhe_params)
        # Decrypt encrypted data using NTRU
        account_data = ntru.decrypt(encrypted_data, self.ntru_params)
        return account_data
