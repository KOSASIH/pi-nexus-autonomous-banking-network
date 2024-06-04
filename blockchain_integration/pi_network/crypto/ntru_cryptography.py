import ntru


class NTRUCryptography:
    def __init__(self, private_key, public_key):
        self.private_key = private_key
        self.public_key = public_key

    def encrypt(self, message):
        return ntru.encrypt(message, self.public_key)

    def decrypt(self, ciphertext):
        return ntru.decrypt(ciphertext, self.private_key)


private_key, public_key = ntru.keygen(701, 613, 11)
ntru_cryptography = NTRUCryptography(private_key, public_key)

message = "Hello, Pi Network!"
ciphertext = ntru_cryptography.encrypt(message)
print("Ciphertext:", ciphertext)

plaintext = ntru_cryptography.decrypt(ciphertext)
print("Plaintext:", plaintext)
