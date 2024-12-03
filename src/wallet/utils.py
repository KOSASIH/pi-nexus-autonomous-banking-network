import json

def save_to_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)
    print(f"Data saved to {filename}.")

def load_from_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f"Data loaded from {filename}.")
    return data

def generate_unique_id():
    return secrets.token_hex(16)  # Generate a unique transaction ID
