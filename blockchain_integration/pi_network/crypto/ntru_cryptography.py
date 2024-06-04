import ntru

# Generate NTRU key pair
private_key, public_key = ntru.keygen(701, 613, 11)

# Encrypt a message
ciphertext = ntru.encrypt("Hello, Pi Network!", public_key)

# Decrypt the message
plaintext = ntru.decrypt(ciphertext, private_key)
