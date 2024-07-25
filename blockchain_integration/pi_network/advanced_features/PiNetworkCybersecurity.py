# Importing necessary libraries
import hashlib

# Class for cybersecurity
class PiNetworkCybersecurity:
    def __init__(self):
        self.hash_function = hashlib.sha256

    # Function to hash data
    def hash_data(self, data):
        return self.hash_function(data.encode()).hexdigest()

    # Function to verify data integrity
    def verify_data_integrity(self, data, expected_hash):
        actual_hash = self.hash_data(data)
        return actual_hash == expected_hash

# Example usage
cs = PiNetworkCybersecurity()
data = "Hello, World!"
expected_hash = "315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3"
actual_hash = cs.hash_data(data)
print(cs.verify_data_integrity(data, expected_hash))
