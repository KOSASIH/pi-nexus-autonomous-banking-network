import re
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


class FakeAccount:
    def __init__(self, address):
        self.address = address
        self.signed_transaction = None

    def sign_transaction(self, transaction):
        self.signed_transaction = transaction
        return SimpleNamespace(raw_transaction=b"\xab" * 32)


class FakeAccountManager:
    def __init__(self):
        self._accounts = {}

    def from_key(self, private_key_hex):
        if private_key_hex not in self._accounts:
            self._accounts[private_key_hex] = FakeAccount(SENDER_ADDRESS)
        return self._accounts[private_key_hex]

    def last_account(self):
        return self._accounts[list(self._accounts.keys())[-1]]


class FakeEth:
    def __init__(
        self,
        *,
        chain_id=1,
        fee_market=True,
        gas_estimate=21000,
        gas_price=50_000_000_000,
        base_fee=25_000_000_000,
        priority_fee=2_000_000_000,
    ):
        self.chain_id = chain_id
        self.fee_market = fee_market
        self.gas_estimate = gas_estimate
        self.gas_price = gas_price
        self.base_fee = base_fee
        self.priority_fee = priority_fee
        self.account = FakeAccountManager()
        self.last_sent_raw = None

    def get_transaction_count(self, address):
        return 7

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
            raise AttributeError("max_priority_fee is not supported by this node")
        return self.priority_fee

    def send_raw_transaction(self, raw_transaction):
        self.last_sent_raw = raw_transaction
        return _SEND_HASH


class FakeWeb3:
    def __init__(self, **kwargs):
        self.eth = FakeEth(**kwargs)

    def to_hex(self, value):
        return "0x" + value.hex()

    def is_address(self, value):
        return isinstance(value, str) and bool(_ADDRESS_RE.match(value))

    def to_checksum_address(self, value):
        address = value[2:] if value.startswith("0x") else value
        return "0x" + address.lower()


class TestSecureGenerateKeypair(unittest.TestCase):
    def test_returns_public_and_private_pem(self):
        private_pem, public_pem = secure_generate_keypair()
        self.assertIsInstance(private_pem, bytes)
        self.assertIsInstance(public_pem, bytes)
        self.assertIn(b"BEGIN PRIVATE KEY", private_pem)
        self.assertIn(b"BEGIN PUBLIC KEY", public_pem)

    def test_accepts_custom_key_size(self):
        private_pem, public_pem = secure_generate_keypair(key_size=2048)
        self.assertIn(b"BEGIN PRIVATE KEY", private_pem)
        self.assertIn(b"BEGIN PUBLIC KEY", public_pem)

    def test_password_encrypts_private_key(self):
        private_pem, public_pem = secure_generate_keypair(password=b"hunter2")
        self.assertIn(b"BEGIN ENCRYPTED PRIVATE KEY", private_pem)
        self.assertIn(b"BEGIN PUBLIC KEY", public_pem)

    def test_rejects_weak_key_size(self):
        with self.assertRaises(ValueError):
            secure_generate_keypair(key_size=1024)

    def test_rejects_boolean_key_size(self):
        with self.assertRaises(ValueError):
            secure_generate_keypair(key_size=True)


class TestSecureSendTransactionValidation(unittest.TestCase):
    def setUp(self):
        self.web3 = FakeWeb3()

    def test_rejects_malformed_private_key(self):
        for bad_key in ("0x1234", "nothex" * 10, "0x" + "zz" * 32, 123):
            with self.assertRaises(ValueError):
                secure_send_transaction(self.web3, bad_key, RECIPIENT_ADDRESS, 1)

    def test_rejects_malformed_address(self):
        for bad_address in ("0x1234", "0x" + "zz" * 20, None):
            with self.assertRaises(ValueError):
                secure_send_transaction(self.web3, VALID_PRIVATE_KEY, bad_address, 1)

    def test_rejects_invalid_value(self):
        for bad_value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                secure_send_transaction(self.web3, VALID_PRIVATE_KEY, RECIPIENT_ADDRESS, bad_value)


class TestSecureSendTransactionBehavior(unittest.TestCase):
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
        result, _ = self.send(self.web3)
        self.assertEqual(result, _SEND_HASH_HEX)

    def test_broadcasts_raw_transaction(self):
        self.send(self.web3)
        self.assertEqual(self.web3.eth.last_sent_raw, b"\xab" * 32)

    def test_normalizes_unprefixed_private_key(self):
        self.send(self.web3, private_key=VALID_PRIVATE_KEY_NO_PREFIX)
        self.assertEqual(self.web3.eth.account.last_account().address, SENDER_ADDRESS)

    def test_normalizes_recipient_address(self):
        _, account = self.send(
            self.web3, recipient="0x" + ("33" * 20).upper()
        )
        self.assertEqual(account.signed_transaction["to"], RECIPIENT_ADDRESS)

    def test_builds_eip1559_transaction(self):
        _, account = self.send(self.web3, value=42)
        transaction = account.signed_transaction
        self.assertEqual(transaction["type"], "0x2")
        self.assertEqual(transaction["chainId"], 1)
        self.assertEqual(transaction["value"], 42)
        self.assertEqual(transaction["gas"], int(21000 * 1.1) + 1000)
        self.assertEqual(transaction["maxPriorityFeePerGas"], 2_000_000_000)
        self.assertEqual(transaction["maxFeePerGas"], 25_000_000_000 * 2 + 2_000_000_000)

    def test_falls_back_to_legacy_on_pre_london_chain(self):
        web3 = FakeWeb3(fee_market=False)
        _, account = self.send(web3)
        transaction = account.signed_transaction
        self.assertEqual(transaction["type"], "0x0")
        self.assertEqual(transaction["gasPrice"], 50_000_000_000)
        self.assertNotIn("maxFeePerGas", transaction)

    def test_falls_back_gas_when_estimation_fails(self):
        web3 = FakeWeb3(gas_estimate=None)
        _, account = self.send(web3)
        self.assertEqual(account.signed_transaction["gas"], 21000)


if __name__ == "__main__":
    unittest.main()