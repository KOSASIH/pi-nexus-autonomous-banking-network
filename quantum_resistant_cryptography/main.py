import os
from quantum_resistant_cryptography.lattice_cryptography import LatticeCryptography
from quantum_resistant_cryptography.hash_based_signatures import HashBasedSignatures

def main():
    # Generate a lattice-based cryptographic key pair
    lattice_cryptography = LatticeCryptography(dimension=256, modulus=65537)
    private_key = lattice_cryptography.private_key
    public_key = lattice_cryptography.public_key

    # Serialize the public key to a file
    with open('public_key.pem', 'wb') as f:
        f.write(lattice_cryptography.serialize_public_key())

    # Generate a hash-based signature key pair
    hash_based_signatures = HashBasedSignatures(hash_function=hashlib.sha256)
    private_key = hash_based_signatures.generate_private_key()
    public_key = hash_based_signatures.generate_public_key()

    # Serialize the public key to a file
    with open('public_key.pem', 'wb') as f:
        f.write(hash_based_signatures.serialize_public_key(public_key))

    # Sign a message using the hash-based signature scheme
    message = b'Hello, World!'
    signature = hash_based_signatures.sign(message, private_key)

    # Verify the signature using the public key
    hash_based_signatures.verify(message, signature, public_key)

if __name__ == '__main__':
    main()
