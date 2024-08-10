import hashlib

def hash_data(data):
  hashed_data = hashlib.sha256(data.encode()).hexdigest()
  return hashed_data
