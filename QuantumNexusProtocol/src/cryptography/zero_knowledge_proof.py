from hashlib import sha256
import random

class ZeroKnowledgeProof:
    def __init__(self, secret):
        self.secret = secret

    def commit(self):
        self.random_value = random.randint(1, 100)
        self.commitment = sha256(f"{self.secret}{self.random_value}".encode()).hexdigest()
        return self.commitment

    def verify(self, challenge, response):
        return sha256(f"{self.secret}{response}".encode()).hexdigest() == challenge

# Example usage
if __name__ == "__main__":
    secret = "my_secret"
    zk_proof = ZeroKnowledgeProof(secret)
    commitment = zk_proof.commit()
    print(f"Commitment: {commitment}")
    challenge = commitment  # In a real scenario, this would be sent to the verifier
    response = zk_proof.random_value  # The prover sends back the random value
    is_valid = zk_proof.verify(challenge, response)
    print(f"Verification: {is_valid}")
