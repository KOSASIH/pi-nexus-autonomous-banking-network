import ntru

# Generate NTRU key pair
private_key, public_key = ntru.generate_keypair(security_level=128)

# Encrypt data using NTRU
def encrypt_data(plain_text, public_key):
    cipher_text = ntru.encrypt(plain_text, public_key)
    return cipher_text
