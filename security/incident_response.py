"""Signed incident reporting and response verification.

`SecurityIncidentResponse` turns arbitrary incident payloads into a signed
response object and verifies later claim/response pairs against the same
oracle, so the network can prove which incidents were acknowledged and by
whom. The signed payload binds a SHA-256 digest of the reported incident
data, so an acknowledgement for one incident cannot be repurposed as proof
for a different one.
"""

import hashlib
import json
import uuid


class SecurityIncidentResponse:
    """Report and verify incidents through a signed response channel."""

    def __init__(self, oracle_api):
        self.oracle_api = oracle_api

    @staticmethod
    def _canonical(value):
        """Return the canonical form used for signing and verification."""
        if not isinstance(value, str):
            return json.dumps(value, sort_keys=True)
        return value

    def _digest(self, value):
        """Return the SHA-256 hex digest of the canonical ``value``."""
        canonical = self._canonical(value)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def report_incident(self, incident_data):
        """Produce a signed acknowledgement for the reported incident.

        The SHA-256 digest of the canonicalized ``incident_data`` is bound
        into the response before signing, so the acknowledgement cannot be
        presented as proof for any other payload.

        Args:
            incident_data: incident payload to acknowledge.

        Returns:
            ``{"response": {...}, "signature": {...}}`` where the response
            carries the bound ``incident_sha256`` digest.

        Raises:
            RuntimeError: if the oracle cannot produce a signature (e.g. no
                keypair is loaded), so a ``reported`` acknowledgement is
                never emitted without a valid signature.
        """
        response = {
            "incident_id": uuid.uuid4().hex,
            "status": "reported",
            "incident_sha256": self._digest(incident_data),
        }
        response_canonical = self._canonical(response)
        signature = self.oracle_api.sign_data(response_canonical)
        if not isinstance(signature, dict) or "signature" not in signature:
            message = "security oracle could not sign the incident response"
            raise RuntimeError(message)
        return {"response": response, "signature": signature}

    def verify_response(self, response, signature, incident_data=None):
        """Verify a previously issued signed response.

        The whole signed response - including the bound ``incident_sha256``
        digest - must verify against ``signature``. When ``incident_data`` is
        supplied it must hash to the same digest, so the response can also
        be checked against the original payload rather than only itself.

        Args:
            response: the unsigned response object from ``report_incident``.
            signature: the hex signature string returned by the oracle API.
            incident_data: optional original payload to pin the digest to.

        Returns:
            ``{"message": ...}`` describing the verification result.
        """
        if not isinstance(response, dict) or "incident_sha256" not in response:
            return {"message": "Invalid response signature"}
        if incident_data is not None:
            if self._digest(incident_data) != response["incident_sha256"]:
                return {"message": "Invalid response signature"}
        response_canonical = self._canonical(response)
        verification = self.oracle_api.verify_signature(
            response_canonical, signature
        )
        if verification.get("message") == "Signature verified successfully":
            return {"message": "Response verified successfully"}
        return {"message": "Invalid response signature"}
