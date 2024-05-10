import base64
import hashlib
import hmac


def sha256(message: str) -> str:
    """
    Returns the SHA-256 hash of the given message.
    """
    return hashlib.sha256(message.encode()).hexdigest()


def hmac_sha256(message: str, key: str) -> str:
    """
    Returns the HMAC-SHA-256 hash of the given message and key.
    """
    return hmac.new(key.encode(), message.encode(), hashlib.sha256).hexdigest()


def base64_encode(data: bytes) -> str:
    """
    Returns the base64 encoding of the given data.
    """
    return base64.b64encode(data).decode()


def base64_decode(data: str) -> bytes:
    """
    Returns the base64 decoding of the given data.
    """
    return base64.b64decode(data.encode())
