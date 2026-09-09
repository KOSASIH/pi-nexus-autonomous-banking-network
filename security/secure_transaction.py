"""Secure transaction primitives for the Pi-Nexus Autonomous Banking Network.

This module is the single source of truth for the two public helpers that the
security package exposes to sign and broadcast transfers, and to provision
asymmetric keypairs:

* :func:`secure_send_transaction` - validate inputs, build an EIP-1559 (or
  legacy) transfer, sign it locally and broadcast it.
* :func:`secure_generate_keypair` - generate an RSA keypair and serialize it
  to PEM (PKCS8 private key / SubjectPublicKeyInfo public key).

The module intentionally imports neither ``web3`` nor ``cryptography`` at
import time: the ``web3`` object is duck-typed (only ``.eth`` and conversion
helpers are used) and the cryptography dependency is only required when a
keypair is actually generated. This keeps the module importable in constrained
environments and makes the unit tests runnable fully offline.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Matches a 32-byte (64 hex character) private key with an optional 0x prefix.
_PRIVATE_KEY_RE = re.compile(r"^(?:0x)?[0-9a-fA-F]{64}$")
# Matches a 20-byte (40 hex digit) Ether address with optional 0x prefix.
_ADDRESS_RE = re.compile(r"^(?:0x)?[0-9a-fA-F]{40}$")

_MIN_KEY_SIZE = 2048
_MIN_TRANSFER_GAS = 21000
_GAS_BUFFER_FACTOR = 1.1
_GAS_BUFFER_WEI = 1000
# 1 gwei; only used when no fee-history API is available.
_FEE_PRIORITY_FALLBACK = 1_000_000_000


def _normalize_private_key(private_key_hex: str) -> str:
    """Validate and normalize a hex private key to the ``0x``-prefixed form.

    Args:
        private_key_hex: hex-encoded 32-byte private key, with or without the
            ``0x`` prefix.

    Returns:
        The normalized, ``0x``-prefixed hex string suitable for
        ``web3.eth.account.from_key``.

    Raises:
        ValueError: if the value is not a hex-encoded 32-byte key.
    """
    is_key = isinstance(private_key_hex, str) and bool(
        _PRIVATE_KEY_RE.match(private_key_hex)
    )
    if not is_key:
        raise ValueError("private_key_hex must be a hex-encoded 32-byte key")
    if private_key_hex.startswith("0x"):
        normalized = private_key_hex[2:]
    else:
        normalized = private_key_hex
    return "0x" + normalized.lower()


def _validate_address(web3, to_address: str) -> str:
    """Validate and checksum-normalize a destination address.

    Args:
        web3: an initialized web3.Web3 instance (duck-typed).
        to_address: hex-encoded Ethereum address.

    Returns:
        Checksummed address.

    Raises:
        ValueError: if the address is malformed.
    """
    if not isinstance(to_address, str) or not _ADDRESS_RE.match(to_address):
        raise ValueError("to_address must be a 40-character Ethereum address")

    is_address = getattr(web3, "is_address", None)
    if callable(is_address) and not is_address(to_address):
        raise ValueError("to_address is not a valid Ethereum address")

    raw = to_address[2:] if to_address.startswith("0x") else to_address
    if raw != raw.lower() and raw != raw.upper():
        # Mixed-case addresses are only trustworthy when they carry a valid
        # EIP-55 checksum; accepting them silently would let a one-character
        # typo change the destination account.
        is_checksum_address = getattr(web3, "is_checksum_address", None)
        if callable(is_checksum_address):
            if not is_checksum_address(to_address):
                raise ValueError("to_address has an invalid EIP-55 checksum")

    to_checksum_address = getattr(web3, "to_checksum_address", None)
    if callable(to_checksum_address):
        try:
            return to_checksum_address(to_address)
        except Exception:
            pass
    return "0x" + raw.lower()


def _validate_value(value_wei: int) -> int:
    """Validate a transfer value expressed in wei.

    Args:
        value_wei: non-negative integer amount in wei.

    Returns:
        The validated integer value.

    Raises:
        ValueError: if the value is not a non-negative integer.
    """
    if isinstance(value_wei, bool) or not isinstance(value_wei, int):
        raise ValueError("value_wei must be an integer number of wei")
    if value_wei < 0:
        raise ValueError("value_wei must be non-negative")
    return value_wei


def _estimate_gas(
    web3, from_address: str, to_address: str, value_wei: int
) -> int:
    """Estimate gas for a transfer, falling back to the intrinsic cost.

    Some node states (e.g. blank senders, exotic chain state) make
    ``estimate_gas`` raise even though the transfer is valid. In that case we
    return the intrinsic cost of a plain value transfer instead of failing the
    whole payment path. Failure modes that mean the transfer itself would
    revert or exceed available funds are re-raised so the payment is not
    signed and broadcast with too little gas.

    Returns:
        A gas amount with a small buffer, never below ``_MIN_TRANSFER_GAS``.

    Raises:
        RuntimeError: if gas estimation fails in a way that puts the payment
            itself at risk (contract recipient or RPC/client instability).
    """
    is_contract = False
    try:
        code = web3.eth.get_code(to_address)
        is_contract = code not in (None, b"", "0x")
    except Exception:
        is_contract = False

    try:
        estimate = web3.eth.estimate_gas(
            {"from": from_address, "to": to_address, "value": value_wei}
        )
        gas = int(estimate)
    except Exception as exc:
        # A plain EOA-to-EOA value transfer cannot revert, so an estimation
        # failure there is observational (blank sender, RPC hiccup) and the
        # intrinsic 21000-gas cost is safe. Contract recipients routinely
        # need more than that - signing with 21000 would burn fees on a
        # failing transfer, so surface the estimation error instead.
        if is_contract:
            raise RuntimeError(
                "gas estimation failed for a contract recipient"
            ) from exc
        return _MIN_TRANSFER_GAS
    buffered = int(gas * _GAS_BUFFER_FACTOR) + _GAS_BUFFER_WEI
    return max(buffered, _MIN_TRANSFER_GAS)


def secure_send_transaction(
    web3, private_key_hex: str, to_address: str, value_wei: int
) -> str:
    """Securely sign and broadcast a value transfer, returning its hash.

    The transaction is built with dynamic EIP-1559 fee fields when the chain
    supports them; otherwise it falls back to a legacy ``gasPrice``
    transaction. All inputs are validated up front so that malformed keys,
    destinations or values never reach the signing path.

    Args:
        web3: an initialized web3.Web3 instance (duck-typed; only ``.eth`` and
            the conversion helpers are used).
        private_key_hex: hex-encoded 32-byte private key, with or without the
            ``0x`` prefix.
        to_address: hex-encoded destination Ethereum address.
        value_wei: non-negative integer amount in wei.

    Returns:
        The ``0x``-prefixed transaction hash.

    Raises:
        ValueError: if any input is malformed.
    """
    private_key = _normalize_private_key(private_key_hex)
    destination = _validate_address(web3, to_address)
    value = _validate_value(value_wei)

    account = web3.eth.account.from_key(private_key)
    # "pending" avoids reusing a nonce when an earlier send from the same
    # account is still in the mempool, which would silently replace/reject it.
    nonce = web3.eth.get_transaction_count(account.address, "pending")
    gas = _estimate_gas(web3, account.address, destination, value)

    chain_id = getattr(web3.eth, "chain_id", None)

    transaction = {
        "from": account.address,
        "to": destination,
        "value": value,
        "gas": gas,
        "nonce": nonce,
    }
    if chain_id is not None:
        transaction["chainId"] = chain_id

    try:
        fee_history = web3.eth.fee_history(1, "latest")
        base_fee = int(fee_history["baseFeePerGas"][-1])
        priority_fee = getattr(web3.eth, "max_priority_fee", None)
        if priority_fee is None:
            priority_fee = _FEE_PRIORITY_FALLBACK
        priority_fee = int(priority_fee)
        transaction.update(
            {
                "maxFeePerGas": int(base_fee * 2) + priority_fee,
                "maxPriorityFeePerGas": priority_fee,
                "type": "0x2",
            }
        )
    except Exception:
        gas_price = getattr(web3.eth, "gas_price", None)
        if gas_price is not None:
            transaction.update({"gasPrice": int(gas_price), "type": "0x0"})

    signed = account.sign_transaction(transaction)
    raw_transaction = getattr(signed, "raw_transaction", None)
    if raw_transaction is None:
        raw_transaction = getattr(signed, "rawTransaction")
    transaction_hash = web3.eth.send_raw_transaction(raw_transaction)
    return web3.to_hex(transaction_hash)


def secure_generate_keypair(
    key_size: int = 4096, password: Optional[bytes] = None
) -> Tuple[bytes, bytes]:
    """Generate an RSA keypair serialized to PEM.

    The private key is written as PKCS8 and the public key as
    SubjectPublicKeyInfo. Passing ``password`` enables best-available
    PEM encryption for the private key.

    Args:
        key_size: RSA modulus size in bits (default ``4096``).
        password: optional password bytes used to encrypt the private key.

    Returns:
        A ``(private_pem, public_pem)`` tuple of PEM-encoded bytes.

    Raises:
        ValueError: if ``key_size`` is smaller than ``_MIN_KEY_SIZE`` or the
            cryptography library is unavailable.
    """
    is_bad = isinstance(key_size, bool) or not isinstance(key_size, int)
    if is_bad or key_size < _MIN_KEY_SIZE:
        message = "key_size must be at least " + str(_MIN_KEY_SIZE)
        raise ValueError(message)

    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError as exc:  # pragma: no cover - environment dependency
        message = "cryptography required for keypair generation"
        raise ValueError(message) from exc

    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=key_size
    )

    if password is None:
        encryption_algorithm = serialization.NoEncryption()
    else:
        encryption_algorithm = serialization.BestAvailableEncryption(password)

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encryption_algorithm,
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem
