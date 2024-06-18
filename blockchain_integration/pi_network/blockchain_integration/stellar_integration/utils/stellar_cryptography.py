from stellar_sdk import StrKey

def generate_keypair():
    return StrKey.random()

def get_public_key(secret_key):
    return StrKey.encode_public_key(secret_key)

def get_secret_key(public_key):
    return StrKey.decode_ed25519_public_key(public_key)
