import hashlib

def encrypt(data, key):
  encrypted_data = hashlib.sha256(data.encode()).hexdigest()
  return encrypted_data

def decrypt(encrypted_data, key):
  decrypted_data = hashlib.sha256(encrypted_data.encode()).hexdigest()
  return decrypted_data
