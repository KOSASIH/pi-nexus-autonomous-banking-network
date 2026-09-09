"""Backward-compatible facade for the :mod:`security` package.

Implementation lives in :mod:`security.secure_transaction`. This module is
kept so the documented public API (``secure_generate_keypair`` and
``secure_send_transaction``) remains available to callers who target the root
module and so the tree parses cleanly.

Note: while a ``security/`` package exists at the repository root, ``import
security`` resolves to the package; its ``__init__`` re-exports the same two
helpers.
"""

from security.secure_transaction import secure_generate_keypair, secure_send_transaction

__all__ = ["secure_generate_keypair", "secure_send_transaction"]