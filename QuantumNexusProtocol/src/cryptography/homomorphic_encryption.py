from phe import paillier

class HomomorphicEncryption:
    def __init__(self):
        self.public_key, self.private_key = paillier.generate_paillier_keypair()

    def encrypt(self, value):
        return self.public_key.encrypt(value)

    def add_encrypted(self, encrypted_value1, encrypted_value2):
        return encrypted_value1 + encrypted_value2

    def decrypt(self, encrypted_value):
        return self.private_key.decrypt(encrypted_value)

# Example usage
if __name__ == "__main__":
    he = HomomorphicEncryption()
    encrypted_value1 = he.encrypt(10)
    encrypted_value2 = he.encrypt(20)
    encrypted_sum = he.add_encrypted(encrypted_value1, encrypted_value2)
    decrypted_sum = he.decrypt(encrypted_sum)
    print(f"Decrypted Sum: {decrypted_sum}")
