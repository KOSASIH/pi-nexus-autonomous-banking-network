from cryptography.fernet import Fernet

class SecureMPC:
    def __init__(self):
        self.keys = [Fernet.generate_key() for _ in range(3)]
        self.fernets = [Fernet(key) for key in self.keys]

    def encrypt(self, data, party_index):
        return self.fernets[party_index].encrypt(data.encode())

    def decrypt(self, encrypted_data, party_index):
        return self.fernets[party_index].decrypt(encrypted_data).decode()

    def compute_sum(self, encrypted_data):
        decrypted_data = [self.decrypt(data, i) for i, data in enumerate(encrypted_data)]
        return sum(map(int, decrypted_data))

# Example usage
if __name__ == "__main__":
    mpc = SecureMPC()
    data = ["10", "20", "30"]
    encrypted_data = [mpc.encrypt(d, i) for i, d in enumerate(data)]

    total = mpc.compute_sum(encrypted_data)
    print(f"Total Sum: {total}")
