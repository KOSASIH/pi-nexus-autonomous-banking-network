from secretsharing import SecretSharer

class SecureMultipartyComputation:
    def __init__(self, threshold):
        self.threshold = threshold

    def share_secret(self, secret, num_shares):
        return SecretSharer.split_secret(secret, self.threshold, num_shares)

    def reconstruct_secret(self, shares):
        return SecretSharer.recover_secret(shares)

# Example usage
if __name__ == "__main__":
    smpc = SecureMultipartyComputation(threshold=3)
    secret = "Super Secret"
    shares = smpc.share_secret(secret, num_shares=5)
    print(f"Shares: {shares}")
    reconstructed_secret = smpc.reconstruct_secret(shares[:3])  # Using 3 shares
    print(f"Reconstructed Secret: {reconstructed_secret}")
