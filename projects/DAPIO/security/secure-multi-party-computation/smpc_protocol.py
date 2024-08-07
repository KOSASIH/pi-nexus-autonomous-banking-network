import random
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

class SMPCProtocol:
    def __init__(self, parties: list, threshold: int):
        self.parties = parties
        self.threshold = threshold
        self.public_keys = [serialization.load_pem_public_key(party.encode(), backend=default_backend()) for party in parties]

    def share_secret(self, secret: int) -> list:
        shares = []
        for i in range(len(self.parties)):
            share = secret + random.randint(0, 2**256 - 1)
            shares.append(share)
        return shares

    def reconstruct_secret(self, shares: list) -> int:
        secret = 0
        for share in shares:
            secret += share
        return secret % 2**256
