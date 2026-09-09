"""Backward-compatible lazy exports for the ``security`` package.

The package re-exports the signing helpers from
:mod:`security.secure_transaction` and the classic authorization helpers
without importing the ``cryptography`` dependency at load time (PEP 562):
the heavy imports happen only when a symbol is actually accessed, which lets
``import security`` work in constrained environments and keeps the offline
test suite hermetic.
"""

import typing

if typing.TYPE_CHECKING:  # pragma: no cover - typing only
    from .authentication import Authentication
    from .authorization import Authorization
    from .decryption import Decryption
    from .encryption import Encryption
    from .secure_transaction import (
        secure_generate_keypair,
        secure_send_transaction,
    )

__all__ = [
    "Authorization",
    "Authentication",
    "Decryption",
    "Encryption",
    "secure_generate_keypair",
    "secure_send_transaction",
]

_LAZY_EXPORTS = {
    "Authorization": "authorization",
    "Authentication": "authentication",
    "Decryption": "decryption",
    "Encryption": "encryption",
    "secure_generate_keypair": "secure_transaction",
    "secure_send_transaction": "secure_transaction",
}


def __getattr__(name):
    """Resolve lazily exported members on first access (PEP 562)."""
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
