import fhe

# Generate FHE key pair
private_key, public_key = fhe.generate_keypair(security_level=128)

# Encrypt data using FHE
def encrypt_data(plain_text, public_key):
    cipher_text = fhe.encrypt(plain_text, public_key)
    return cipher_text

# Perform computations on encrypted data
def compute_on_encrypted_data(cipher_text, operation):
    result = fhe.compute(cipher_text, operation)
    return result
