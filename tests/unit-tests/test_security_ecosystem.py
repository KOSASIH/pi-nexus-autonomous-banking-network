"""Offline tests for the security ecosystem shipped through the
`security` package: SecurityManager, SecurityOracle, SecurityOracleAPI and
SecurityIncidentResponse.
"""

import os
import tempfile
import unittest

from security import (
    SecurityIncidentResponse,
    SecurityManager,
    SecurityOracle,
    SecurityOracleAPI,
)


class SecurityManagerTests(unittest.TestCase):
    """Offline tests for password-derived encryption and authentication."""

    def test_no_argument_construction(self):
        """A manager can be built without credentials."""
        manager = SecurityManager()
        self.assertIsNone(manager.user_id)

    def test_encrypt_decrypt_roundtrip_with_unicode(self):
        """Encryption and decryption round-trip unicode payloads."""
        manager = SecurityManager("alice", "s3cret")
        payload = "sensitive-data-\u00e9\u4e2d\u6587"
        ciphertext = manager.encrypt(payload)
        self.assertNotIn(payload, ciphertext.decode("utf-8", errors="replace"))
        self.assertEqual(manager.decrypt(ciphertext), payload)

    def test_tampered_ciphertext_is_rejected(self):
        """A single flipped byte invalidates authenticated decryption."""
        manager = SecurityManager("alice", "s3cret")
        ciphertext = bytearray(manager.encrypt("Sensitive Data"))
        ciphertext[-1] ^= 0xFF
        with self.assertRaises(Exception):
            manager.decrypt(bytes(ciphertext))

    def test_authenticate_user_success_and_failure(self):
        """The right password authenticates, the wrong one does not."""
        manager = SecurityManager("alice", "s3cret")
        self.assertTrue(manager.authenticate_user("s3cret"))
        self.assertFalse(manager.authenticate_user("wrong"))

    def test_authenticate_user_without_password_is_safe(self):
        """Authentication with no stored password never succeeds."""
        manager = SecurityManager("alice")
        self.assertFalse(manager.authenticate_user("anything"))

    def test_encryption_uses_fresh_nonce_per_call(self):
        """Repeated encryption of the same data yields distinct outputs."""
        manager = SecurityManager("alice", "s3cret")
        first = manager.encrypt("same")
        second = manager.encrypt("same")
        self.assertNotEqual(first, second)
        self.assertEqual(manager.decrypt(first), manager.decrypt(second))

    def test_salt_differs_between_managers(self):
        """Identical credentials produce independent manager keys."""
        first = SecurityManager("alice", "s3cret").derive_key()
        second = SecurityManager("alice", "s3cret").derive_key()
        self.assertNotEqual(first, second)


class SecurityOracleTests(unittest.TestCase):
    """Offline tests for RSA key generation, signing and verification."""

    def test_keypair_generation_makes_oracle_ready(self):
        """Generating keys makes the oracle ready to sign and verify."""
        oracle = SecurityOracle()
        self.assertFalse(oracle.ready)
        oracle.generate_keys()
        self.assertTrue(oracle.ready)

    def test_sign_and_verify_roundtrip(self):
        """A signature verifies against the signed data."""
        oracle = SecurityOracle()
        oracle.generate_keys()
        signature = oracle.sign_data("payload")
        self.assertTrue(oracle.verify_signature("payload", signature))

    def test_verify_rejects_tampered_data(self):
        """A signature fails against different data."""
        oracle = SecurityOracle()
        oracle.generate_keys()
        signature = oracle.sign_data("payload")
        self.assertFalse(oracle.verify_signature("forged", signature))

    def test_verify_without_public_key_is_false(self):
        """Verification without a public key reports False."""
        oracle = SecurityOracle()
        self.assertFalse(oracle.verify_signature("payload", b"deadbeef"))

    def test_sign_without_private_key_raises(self):
        """Signing without a private key raises ValueError."""
        oracle = SecurityOracle()
        with self.assertRaises(ValueError):
            oracle.sign_data("payload")

    def test_missing_key_files_degrades_gracefully(self):
        """Missing key files leave the oracle unready but usable."""
        missing = os.path.join(tempfile.gettempdir(), "missing-key-keys.pem")
        oracle = SecurityOracle(missing, missing)
        self.assertFalse(oracle.ready)
        self.assertFalse(oracle.verify_signature("payload", b"deadbeef"))

    def test_generate_keys_persists_pem_files(self):
        """Generated keys persist to PEM files and reload cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            private_path = os.path.join(tmpdir, "private.pem")
            public_path = os.path.join(tmpdir, "public.pub")
            oracle = SecurityOracle(private_path, public_path)
            oracle.generate_keys()
            self.assertTrue(os.path.exists(private_path))
            self.assertTrue(os.path.exists(public_path))
            reloaded = SecurityOracle(private_path, public_path)
            self.assertTrue(reloaded.ready)

    @unittest.skipIf(os.name == "nt", "file modes differ on Windows")
    def test_private_key_file_is_not_world_readable(self):
        """Generated private keys are restricted to their owner (0600)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            private_path = os.path.join(tmpdir, "private.pem")
            public_path = os.path.join(tmpdir, "public.pub")
            oracle = SecurityOracle(private_path, public_path)
            oracle.generate_keys()
            private_mode = os.stat(private_path).st_mode & 0o777
            public_mode = os.stat(public_path).st_mode & 0o777
            self.assertEqual(private_mode, 0o600)
            self.assertNotEqual(public_mode, 0o600)


class SecurityOracleAPITests(unittest.TestCase):
    """Offline tests for the dictionary-shaped oracle API wrapper."""

    def setUp(self):
        """Prepare an in-memory oracle with a fresh keypair."""
        oracle = SecurityOracle()
        oracle.generate_keys()
        self.api = SecurityOracleAPI(oracle)

    def test_sign_data_returns_hex_signature(self):
        """Signature output is a valid hex string."""
        result = self.api.sign_data("payload")
        self.assertIn("signature", result)
        signature_hex = result["signature"]
        self.assertEqual(len(signature_hex) % 2, 0)
        self.assertTrue(bytes.fromhex(signature_hex))

    def test_sign_data_verifiable_through_api(self):
        """A hex signature verifies through the API wrapper."""
        signed = self.api.sign_data("payload")
        verified = self.api.verify_signature(
            "payload", signed["signature"]
        )
        self.assertEqual(
            verified["message"], "Signature verified successfully"
        )

    def test_verify_signature_rejects_forged(self):
        """A crafted signature is rejected by the API wrapper."""
        verified = self.api.verify_signature("payload", "ff" * 64)
        self.assertEqual(verified["message"], "Invalid signature")

    def test_verify_signature_handles_malformed_hex(self):
        """Malformed hex input yields Invalid signature, not an error."""
        for bad_signature in ("not-hex!", "zz", "", 123, None):
            verified = self.api.verify_signature("payload", bad_signature)
            self.assertEqual(verified["message"], "Invalid signature")

    def test_generate_keys_returns_message(self):
        """The API reports key generation success."""
        result = self.api.generate_keys()
        self.assertEqual(result["message"], "Keys generated successfully")

    def test_sign_data_without_key_returns_error(self):
        """Signing before keys exist returns an error dict."""
        api = SecurityOracleAPI(SecurityOracle())
        self.assertIn("error", api.sign_data("payload"))


class SecurityIncidentResponseTests(unittest.TestCase):
    """Offline tests for signed incident reporting and verification."""

    def setUp(self):
        """Prepare an oracle-backed incident reporter."""
        oracle = SecurityOracle()
        oracle.generate_keys()
        self.api = SecurityOracleAPI(oracle)
        self.incident_response = SecurityIncidentResponse(self.api)

    def test_report_incident_returns_signed_response(self):
        """Reporting an incident returns status and a signature."""
        reported = self.incident_response.report_incident({"source": "node-1"})
        self.assertEqual(reported["response"]["status"], "reported")
        self.assertIn("incident_id", reported["response"])
        self.assertIn("signature", reported["signature"])

    def test_response_is_bound_to_the_incident_payload(self):
        """The signed response carries a digest bound to the payload."""
        first = self.incident_response.report_incident({"source": "node-1"})
        second = self.incident_response.report_incident({"source": "node-2"})
        self.assertIn("incident_sha256", first["response"])
        self.assertNotEqual(
            first["response"]["incident_sha256"],
            second["response"]["incident_sha256"],
        )
        verification = self.incident_response.verify_response(
            first["response"],
            second["signature"]["signature"],
            incident_data={"source": "node-1"},
        )
        self.assertEqual(
            verification["message"], "Invalid response signature"
        )

    def test_verify_response_can_pin_the_original_payload(self):
        """Supplying the original payload tightens verification."""
        reported = self.incident_response.report_incident({"source": "node-1"})
        matching = self.incident_response.verify_response(
            reported["response"],
            reported["signature"]["signature"],
            incident_data={"source": "node-1"},
        )
        self.assertEqual(
            matching["message"], "Response verified successfully"
        )
        conflicting = self.incident_response.verify_response(
            reported["response"],
            reported["signature"]["signature"],
            incident_data={"source": "node-2"},
        )
        self.assertEqual(
            conflicting["message"], "Invalid response signature"
        )

    def test_report_incident_without_ready_oracle_raises(self):
        """No reported acknowledgement is emitted when signing fails."""
        api = SecurityOracleAPI(SecurityOracle())
        incident_response = SecurityIncidentResponse(api)
        with self.assertRaises(RuntimeError):
            incident_response.report_incident({"source": "node-1"})

    def test_reported_response_verifies(self):
        """A reported response verifies against its signature."""
        reported = self.incident_response.report_incident({"source": "node-1"})
        verification = self.incident_response.verify_response(
            reported["response"], reported["signature"]["signature"]
        )
        self.assertEqual(
            verification["message"], "Response verified successfully"
        )

    def test_tampered_response_is_rejected(self):
        """A modified response fails verification."""
        reported = self.incident_response.report_incident({"source": "node-1"})
        tampered = dict(reported["response"])
        tampered["status"] = "escalated"
        verification = self.incident_response.verify_response(
            tampered, reported["signature"]["signature"]
        )
        self.assertEqual(
            verification["message"], "Invalid response signature"
        )


if __name__ == "__main__":
    unittest.main()
