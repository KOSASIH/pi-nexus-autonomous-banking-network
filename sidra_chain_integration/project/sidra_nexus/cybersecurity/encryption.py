# cybersecurity/encryption.py
import hashlib

# Hash a password
password = 'your_password'
hashed_password = hashlib.sha256(password.encode()).hexdigest()
