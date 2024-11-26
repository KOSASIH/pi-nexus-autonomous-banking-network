import hashlib

class SecureHashing:
    @staticmethod
    def hash_data(data):
        return hashlib.sha256(data.encode()).hexdigest()

# Example usage
if __name__ == "__main__":
    data = "Secure this data!"
    hashed_data = SecureHashing.hash_data(data)
    print(f"Hashed Data: {hashed_data}")
