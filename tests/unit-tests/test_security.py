"""Offline unit tests for :mod:`security.secure_transaction`.

The suite covers keypair generation, input validation, EIP-1559/legacy
transaction building, gas estimation behavior, pending-pool nonce selection
and EIP-55 checksum handling - all against a duck-typed fake ``web3``.
"""

import hashlib
import os
import re
import subprocess
import sys
import unittest
from types import SimpleNamespace

from security import secure_generate_keypair, secure_send_transaction

VALID_PRIVATE_KEY = "0x" + "11" * 32
VALID_PRIVATE_KEY_NO_PREFIX = "11" * 32
SENDER_ADDRESS = "0x" + "22" * 20
RECIPIENT_ADDRESS = "0x" + "33" * 20

_SEND_HASH = bytes([0xCD] * 32)
_SEND_HASH_HEX = "0x" + "cd" * 32

_ADDRESS_RE = re.compile(r"^(?:0x)?[0-9a-fA-F]{40}$")


def eip55_checksum(address_hex):
    """Return the EIP-55 checksummed form of a 40-hex Ethereum address."""
    raw = address_hex[2:] if address_hex.startswith("0x") else address_hex
    hash_hex = hashlib.sha3_256(raw.lower().encode("ascii")).hexdigest()
    return "0x" + "".join(
        char.upper()
        if int(hash_hex[index], 16) >= 8
        else char
        for index, char in enumerate(raw)
    )


def mixed_checksummed_address():
    """Return a deterministic EIP-55 address whose letters span both cases."""
    for head in range(1, 40):
        raw = ("a" * head) + ("b" * (40 - head))
        checksummed = eip55_checksum("0x" + raw)
        letters = [c for c in checksummed[2:] if c.isalpha()]
        has_upper = any(c.isupper() for c in letters)
        has_lower = any(c.islower() for c in letters)
        if has_upper and has_lower:
            return checksummed
    raise AssertionError("no mixed-case checksum address found")


def corrupt_mixed_address(address_hex):
    """Return a still-mixed variant with an invalid EIP-55 checksum."""
    raw = address_hex[2:]
    for index, char in enumerate(raw):
        if char.isalpha():
            flipped = raw[:index] + char.swapcase() + raw[index + 1:]
            candidate = "0x" + flipped
            candidate_valid = candidate == eip55_checksum(candidate)
            if candidate != address_hex and not candidate_valid:
                return candidate
    raise AssertionError("could not produce an invalid mixed-case address")


class FakeAccount:
    """A signed-transaction recording stand-in for an account object."""

    def __init__(self, address):
        self.address = address
        self.signed_transaction = None

    def sign_transaction(self, transaction):
        self.signed_transaction = transaction
        return SimpleNamespace(raw_transaction=b"\xab" * 32)


class FakeAccountManager:
    """An ``eth.account`` stand-in keyed by private key."""

    def __init__(self):
        self._accounts = {}

    def from_key(self, private_key_hex):
        if private_key_hex not in self._accounts:
            self._accounts[private_key_hex] = FakeAccount(SENDER_ADDRESS)
        return self._accounts[private_key_hex]

    def last_account(self):
        return self._accounts[list(self._accounts.keys())[-1]]


class FakeEth:
    """A configurable stand-in for the ``web3.eth`` interface."""

    def __init__(
        self,
        *,
        chain_id=1,
        fee_market=True,
        gas_estimate=21000,
        gas_price=50_000_000_000,
        base_fee=25_000_000_000,
        priority_fee=2_000_000_000,
        contract_code=b"",
    ):
        self.chain_id = chain_id
        self.fee_market = fee_market
        self.gas_estimate = gas_estimate
        self.gas_price = gas_price
        self.base_fee = base_fee
        self.priority_fee = priority_fee
        self.contract_code = contract_code
        self.account = FakeAccountManager()
        self.last_nonce_block = None
        self.last_sent_raw = None

    def get_transaction_count(self, address, block=None):
        """Record the requested block tag and return a fixed nonce."""
        self.last_nonce_block = block
        return 7

    def get_code(self, address):
        """Return the bytecode installed at ``address`` (empty for EOAs)."""
        return self.contract_code

    def estimate_gas(self, payload):
        if self.gas_estimate is None:
            raise ValueError("gas estimation failed")
        return self.gas_estimate

    def fee_history(self, block_count, newest_block):
        if not self.fee_market:
            raise AttributeError("fee_history is not supported by this node")
        return {"baseFeePerGas": [self.base_fee]}

    @property
    def max_priority_fee(self):
        if not self.fee_market:
            raise AttributeError(
                "max_priority_fee is not supported by this node"
            )
        return self.priority_fee

    def send_raw_transaction(self, raw_transaction):
        self.last_sent_raw = raw_transaction
        return _SEND_HASH


class FakeWeb3:
    """A duck-typed stand-in for ``web3.Web3`` with EIP-55 helpers."""

    def __init__(self, **kwargs):
        self.eth = FakeEth(**kwargs)

    def to_hex(self, value):
        return "0x" + value.hex()

    def is_address(self, value):
        return isinstance(value, str) and bool(_ADDRESS_RE.match(value))

    def is_checksum_address(self, value):
        return isinstance(value, str) and value == eip55_checksum(value)

    def to_checksum_address(self, value):
        return eip55_checksum(value)


class TestSecureGenerateKeypair(unittest.TestCase):
    """Coverage for :func:`secure_transaction.secure_generate_keypair`."""

    def test_returns_public_and_private_pem(self):
        """Keypair generation returns both PEM blobs."""
        private_pem, public_pem = secure_generate_keypair()
        self.assertIsInstance(private_pem, bytes)
        self.assertIsInstance(public_pem, bytes)
        self.assertIn(b"BEGIN PRIVATE KEY", private_pem)
        self.assertIn(b"BEGIN PUBLIC KEY", public_pem)

    def test_accepts_custom_key_size(self):
        """A key size of 2048 bits is honored."""
        private_pem, public_pem = secure_generate_keypair(key_size=2048)
        self.assertIn(b"BEGIN PRIVATE KEY", private_pem)
        self.assertIn(b"BEGIN PUBLIC KEY", public_pem)

    def test_password_encrypts_private_key(self):
        """A password produces an encrypted PKCS8 private key."""
        private_pem, public_pem = secure_generate_keypair(password=b"hunter2")
        self.assertIn(b"BEGIN ENCRYPTED PRIVATE KEY", private_pem)
        self.assertIn(b"BEGIN PUBLIC KEY", public_pem)

    def test_rejects_weak_key_size(self):
        """Sub-2048-bit keys are rejected."""
        with self.assertRaises(ValueError):
            secure_generate_keypair(key_size=1024)

    def test_rejects_boolean_key_size(self):
        """Boolean key sizes are rejected as non-integers."""
        with self.assertRaises(ValueError):
            secure_generate_keypair(key_size=True)


class TestSecureSendTransactionValidation(unittest.TestCase):
    """Input validation coverage for :func:`secure_send_transaction`."""

    def setUp(self):
        self.web3 = FakeWeb3()

    def test_rejects_malformed_private_key(self):
        """Malformed private keys never reach the signing path."""
        for bad_key in ("0x1234", "nothex" * 10, "0x" + "zz" * 32, 123):
            with self.assertRaises(ValueError):
                secure_send_transaction(
                    self.web3, bad_key, RECIPIENT_ADDRESS, 1
                )

    def test_rejects_malformed_address(self):
        """Malformed destination addresses are rejected up front."""
        for bad_address in ("0x1234", "0x" + "zz" * 20, None):
            with self.assertRaises(ValueError):
                secure_send_transaction(
                    self.web3, VALID_PRIVATE_KEY, bad_address, 1
                )

    def test_rejects_invalid_value(self):
        """Negative, fractional and boolean amounts are rejected."""
        for bad_value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                secure_send_transaction(
                    self.web3,
                    VALID_PRIVATE_KEY,
                    RECIPIENT_ADDRESS,
                    bad_value,
                )


class TestSecureSendTransactionBehavior(unittest.TestCase):
    """Transaction building and broadcasting behavior."""

    def setUp(self):
        self.web3 = FakeWeb3()

    def send(self, web3, **kwargs):
        return secure_send_transaction(
            web3,
            kwargs.get("private_key", VALID_PRIVATE_KEY),
            kwargs.get("recipient", RECIPIENT_ADDRESS),
            kwargs.get("value", 10**18),
        ), web3.eth.account.last_account()

    def test_returns_hex_transaction_hash(self):
        """The broadcast returns a hex transaction hash."""
        result, _ = self.send(self.web3)
        self.assertEqual(result, _SEND_HASH_HEX)

    def test_broadcasts_raw_transaction(self):
        """The signed raw transaction reaches the node."""
        self.send(self.web3)
        self.assertEqual(self.web3.eth.last_sent_raw, b"\xab" * 32)

    def test_normalizes_unprefixed_private_key(self):
        """A key without the 0x prefix is normalized to the same account."""
        self.send(self.web3, private_key=VALID_PRIVATE_KEY_NO_PREFIX)
        self.assertEqual(
            self.web3.eth.account.last_account().address, SENDER_ADDRESS
        )

    def test_normalizes_recipient_address(self):
        """An all-uppercase address is accepted and checksummed."""
        _, account = self.send(
            self.web3, recipient="0x" + ("33" * 20).upper()
        )
        self.assertEqual(account.signed_transaction["to"], RECIPIENT_ADDRESS)

    def test_builds_eip1559_transaction(self):
        """Fee-market chains receive a typed EIP-1559 transaction."""
        _, account = self.send(self.web3, value=42)
        transaction = account.signed_transaction
        self.assertEqual(transaction["type"], "0x2")
        self.assertEqual(transaction["chainId"], 1)
        self.assertEqual(transaction["value"], 42)
        self.assertEqual(transaction["gas"], int(21000 * 1.1) + 1000)
        self.assertEqual(transaction["maxPriorityFeePerGas"], 2_000_000_000)
        self.assertEqual(
            transaction["maxFeePerGas"],
            25_000_000_000 * 2 + 2_000_000_000,
        )

    def test_falls_back_to_legacy_on_pre_london_chain(self):
        """Pre-London chains receive a legacy gasPrice transaction."""
        web3 = FakeWeb3(fee_market=False)
        _, account = self.send(web3)
        transaction = account.signed_transaction
        self.assertEqual(transaction["type"], "0x0")
        self.assertEqual(transaction["gasPrice"], 50_000_000_000)
        self.assertNotIn("maxFeePerGas", transaction)

    def test_falls_back_gas_when_estimation_fails_for_eoa(self):
        """EOA recipients tolerate observational estimation failures."""
        web3 = FakeWeb3(gas_estimate=None)
        _, account = self.send(web3)
        self.assertEqual(account.signed_transaction["gas"], 21000)

    def test_contract_estimation_failure_raises(self):
        """Contract recipients surface estimation failures instead of 21000."""
        web3 = FakeWeb3(gas_estimate=None, contract_code=b"\x00")
        with self.assertRaises(RuntimeError):
            self.send(web3)

    def test_estimation_uses_pending_pool_nonce(self):
        """Nonces are read from the pending pool to avoid reuse."""
        self.send(self.web3)
        self.assertEqual(self.web3.eth.last_nonce_block, "pending")

    def test_rejects_mixed_case_address_with_bad_checksum(self):
        """Reject mixed-case addresses with a bad EIP-55 checksum."""
        corrupted = corrupt_mixed_address(mixed_checksummed_address())
        with self.assertRaises(ValueError):
            secure_send_transaction(self.web3, VALID_PRIVATE_KEY, corrupted, 1)

    def test_rejects_mixed_case_address_without_provider_support(self):
        """Mixed case is unsafe when the provider knows no checksum."""

        class MinimalWeb3(FakeWeb3):
            def is_checksum_address(self, value):
                return None

        with self.assertRaises(ValueError):
            secure_send_transaction(
                MinimalWeb3(),
                VALID_PRIVATE_KEY,
                mixed_checksummed_address(),
                1,
            )

    def test_accepts_valid_checksummed_address(self):
        """A valid EIP-55 checksummed address is passed through unchanged."""
        checksummed = mixed_checksummed_address()
        _, account = self.send(self.web3, recipient=checksummed)
        self.assertEqual(account.signed_transaction["to"], checksummed)


class TestLazyImports(unittest.TestCase):
    """Package import must not require the optional cryptography dependency."""

    def test_package_imports_without_cryptography(self):
        """``from security import ...`` works with cryptography blocked."""
        probe = os.path.dirname(os.path.abspath(__file__))
        while not os.path.isdir(os.path.join(probe, "security")):
            probe = os.path.dirname(probe)
        script = (
            "import sys\n"
            "sys.modules['cryptography'] = None\n"
            "import security\n"
            "secure_generate_keypair = security.secure_generate_keypair\n"
            "tx_send = getattr(security, 'secure_send_transaction')\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=probe,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
