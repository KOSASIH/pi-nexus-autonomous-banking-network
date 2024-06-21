# Identity Manager for Pi Network
import hashlib
from pi_network.core.identity import Identity

class IdentityManager:
    def __init__(self, network_id, node_id):
        self.network_id = network_id
        self.node_id = node_id
        self.identity_registry = {}

    def create_identity(self, public_key: str, metadata: dict) -> str:
        # Create new identity and return identity ID
        identity_hash = hashlib.sha256(public_key.encode()).hexdigest()
        self.identity_registry[identity_hash] = Identity(public_key, metadata)
        return identity_hash

    def get_identity(self, identity_id: str) -> Identity:
        # Return identity by ID
        return self.identity_registry.get(identity_id)

    def update_identity(self, identity_id: str, metadata: dict) -> bool:
        # Update identity metadata
        identity = self.identity_registry.get(identity_id)
        if identity:
            identity.metadata = metadata
            return True
        return False
