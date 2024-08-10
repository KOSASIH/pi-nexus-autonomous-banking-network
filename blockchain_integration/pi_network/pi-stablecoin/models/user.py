import hashlib

class User:
  def __init__(self, username, password):
    self.username = username
    self.password = self.hash_password(password)

  def hash_password(self, password):
    return hashlib.sha256(password.encode()).hexdigest()
