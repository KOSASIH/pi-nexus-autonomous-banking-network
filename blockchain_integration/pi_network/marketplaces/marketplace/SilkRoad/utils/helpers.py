# utils/helpers.py

import hashlib
import secrets
import string


def generate_random_string(length: int = 10) -> str:
    """
    Generate a random string of given length.
    """
    letters = string.ascii_lowercase
    return "".join(secrets.choice(letters) for _ in range(length))


def hash_password(password: str) -> str:
    """
    Hash a password using SHA-256 algorithm.
    """
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    return hashed_password


def is_authenticated(username: str, hashed_password: str) -> bool:
    """
    Check if the provided username and password match the stored hashed password.
    """
    # In a real-world scenario, you would fetch the hashed password from the database.
    stored_hashed_password = hash_password("my_secret_password")
    return stored_hashed_password == hashed_password


def get_random_color() -> str:
    """
    Generate a random color in hex format.
    """
    return "#{:06x}".format(secrets.randbelow(0xFFFFFF))
