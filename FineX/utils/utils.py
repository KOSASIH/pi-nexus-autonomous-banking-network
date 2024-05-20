import re
import jwt
import datetime
import hashlib

def is_valid_email(email):
    """
    A function to validate email addresses.
    """
    regex = r'^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$'
    return re.search(regex, email) is not None

def is_valid_phone_number(phone_number):
"""
    A function to validate phone numbers.
    """
    regex = r'^\+?[\d\s()-]*$'
    return re.search(regex, phone_number) is not None

def format_currency(amount):
    """
    A function to format currency amounts.
    """
    return f'${amount:,.2f}'

def hash_password(password):
    """
    A function to hash passwords.
    """
    salt = hashlib.sha256(b'salt').digest()
    hashed_password = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000
    )
    return hashed_password

def generate_jwt_token(user_id, secret_key):
    """
    A function to generate JWT tokens.
    """
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
    }
    token = jwt.encode(payload, secret_key, algorithm='HS256')
    return token

def parse_jwt_token(token, secret_key):
    """
    A function to parse JWT tokens.
    """
    try:
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
