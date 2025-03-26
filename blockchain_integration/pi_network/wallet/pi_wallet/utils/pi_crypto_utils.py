import hashlib

import base58
import ecdsa


def generate_private_key():
    sk = ecdsa.SigningKey.generate(curve=ecdsa.NIST256p)
    return sk.to_string().hex()


def generate_public_key(private_key):
    sk = ecdsa.SigningKey.from_string(bytes.fromhex(private_key), curve=ecdsa.NIST256p)
    vk = sk.get_verifying_key()
    return vk.to_string().hex()


def private_key_to_public_key(private_key):
    sk = ecdsa.SigningKey.from_string(bytes.fromhex(private_key), curve=ecdsa.NIST256p)
    vk = sk.get_verifying_key()
    return vk.to_string().hex()


def public_key_to_address(public_key):
    public_key_bytes = bytes.fromhex(public_key)
    public_key_bytes = b"\x04" + public_key_bytes[1:]
    address = base58.b58encode(public_key_bytes)
    return address.decode("utf-8")


def sign_transaction(private_key, transaction):
    sk = ecdsa.SigningKey.from_string(bytes.fromhex(private_key), curve=ecdsa.NIST256p)
    signature = sk.sign(transaction.encode())
    return signature.encode()


def verify_signature(public_key, transaction, signature):
    vk = ecdsa.VerifyingKey.from_string(bytes.fromhex(public_key), curve=ecdsa.NIST256p)
    return vk.verify(signature, transaction.encode())


def hash_transaction(transaction):
    transaction_bytes = json.dumps(transaction, sort_keys=True).encode()
    transaction_hash = hashlib.sha256(transaction_bytes).digest()
    transaction_hash = hashlib.sha256(transaction_hash).digest()
    return transaction_hash
