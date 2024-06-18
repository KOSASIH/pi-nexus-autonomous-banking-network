from stellar_sdk import Server, Network, Keypair

def get_stellar_client(stellar_network: str):
    """Return a Stellar SDK server instance for the specified network"""
    if stellar_network == "testnet":
        server = Server("https://horizon-testnet.stellar.org")
    elif stellar_network == "mainnet":
        server = Server("https://horizon.stellar.org")
    else:
        raise ValueError(f"Invalid Stellar network: {stellar_network}")
    return server

def get_account_sequence(account_id: str, stellar_client):
    """Return the sequence number of the specified Stellar account"""
    account = stellar_client.account(account_id)
    return account.sequence

def get_keypair_from_seed(seed: str):
    """Return a Stellar SDK keypair instance from the specified seed"""
    return Keypair.from_seed(seed)
