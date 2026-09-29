"""Security primitives for the Pi-Nexus Autonomous Banking Network.

The package re-exports the signing, keypair, authorization, incident
response and security-oracle helpers from their implementing modules. Heavy
optional dependencies are imported lazily so that merely importing
``security`` stays cheap and works in constrained environments.
"""

import typing

if typing.TYPE_CHECKING:  # pragma: no cover - typing only
    from .authentication import Authentication
    from .authorization import Authorization
    from .decryption import Decryption
    from .encryption import Encryption
    from .incident_response import SecurityIncidentResponse
    from .secure_transaction import (
        secure_generate_keypair,
        secure_send_transaction,
    )
    from .security_manager import SecurityManager
    from .security_oracle import SecurityOracle, SecurityOracleAPI

__all__ = [
    "Authentication",
    "Authorization",
    "Decryption",
    "Encryption",
    "SecurityIncidentResponse",
    "SecurityManager",
    "SecurityOracle",
    "SecurityOracleAPI",
    "secure_generate_keypair",
    "secure_send_transaction",
]

_LAZY_EXPORTS = {
    "Authentication": "authentication",
    "Authorization": "authorization",
    "Decryption": "decryption",
    "Encryption": "encryption",
    "SecurityIncidentResponse": "incident_response",
    "SecurityManager": "security_manager",
    "SecurityOracle": "security_oracle",
    "SecurityOracleAPI": "security_oracle",
    "secure_generate_keypair": "secure_transaction",
    "secure_send_transaction": "secure_transaction",
}


def __getattr__(name):
    """Resolve public package members on demand (PEP 562)."""
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        module = import_module("." + _LAZY_EXPORTS[name], __name__)
        value = getattr(module, name, None)
        if value is None:
            message = f"{name!r} is not exported by {module!r}"
            raise AttributeError(message)
        globals()[name] = value
        return value
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)


def __dir__():
    """Expose the declared public API for ``help`` and ``dir``."""
    return sorted(set(list(globals().keys()) + __all__))
