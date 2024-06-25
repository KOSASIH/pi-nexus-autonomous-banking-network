import numpy as np
from hss import HSS

class HSSMPC:
    def __init__(self, parties, threshold):
        self.parties = parties
        self.threshold = threshold
        self.hss = HSS()

    def share_secret(self, secret):
        # Share the secret among the parties using HSS
        shares = self.hss.share(secret, self.parties, self.threshold)
        return shares

    def reconstruct_secret(self, shares):
        # Reconstruct the secret from the shares using HSS
        secret = self.hss.reconstruct(shares, self.parties, self.threshold)
        return secret

    def compute_function(self, function, inputs):
        # Compute a function on the shared inputs using HSS
        outputs = self.hss.compute(function, inputs, self.parties, self.threshold)
        return outputs
