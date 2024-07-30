import hashlib

class Wallet:
    def __init__(self):
        self.public_key = hashlib.sha256(os.urandom(32)).hexdigest()
        self.private_key = os.urandom(32)
