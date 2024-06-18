import stellar_sdk

def get_stellar_client(network: str) -> stellar_sdk.Client:
    """Return a Stellar client instance for the specified network"""
    if network == "testnet":
        return stellar_sdk.Client(stellar_sdk.Network.TESTNET)
    elif network == "mainnet":
        return stellar_sdk.Client(stellar_sdk.Network.PUBLIC)
    else:
        raise ValueError("Invalid network specified")

def get_account_sequence(account_id: str, client: stellar_sdk.Client) -> int:
    """Return the sequence number for the specified account"""
    account = client.account(account_id)
    return account.sequence

def get_transaction_fee(client: stellar_sdk.Client) -> int:
    """Return the current transaction fee for the Stellar network"""
    fee_stats = client.fee_stats()
    return fee_stats.fee_charged
