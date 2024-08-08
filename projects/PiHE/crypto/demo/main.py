from crypto.key_manager import KeyManager
from crypto.homomorphic_encryption import HomomorphicEncryption

def main():
    key_manager = KeyManager(key_size=2048, ciphertext_size=2048)
    key_manager.generate_keys()

    he = HomomorphicEncryption(key_manager.get_public_key(), key_manager.get_private_key())

    plaintext1 = 5
    plaintext2 = 3

    ciphertext1 = he.encrypt(plaintext1)
    ciphertext2 = he.encrypt(plaintext2)

    ciphertext_add = he.add(ciphertext1, ciphertext2)
    ciphertext_multiply = he.multiply(ciphertext1, ciphertext2)

    decrypted_add = he.decrypt(ciphertext_add)
    decrypted_multiply = he.decrypt(ciphertext_multiply)

    print(f"Plaintext 1: {plaintext1}")
    print(f"Plaintext 2: {plaintext2}")
    print(f"Encrypted addition: {decrypted_add}")
    print(f"Encrypted multiplication: {decrypted_multiply}")

if __name__ == "__main__":
    main()
