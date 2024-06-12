import uport

# Set up uPort identity management
uport_client = uport.Client()

# Create decentralized identity
def create_identity(did, public_key):
    identity = uport_client.create_identity(did, public_key)
    return identity
