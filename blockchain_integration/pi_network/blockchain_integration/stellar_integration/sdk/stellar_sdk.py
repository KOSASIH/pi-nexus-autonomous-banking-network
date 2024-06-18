import stellar_sdk

class PiNetworkStellarIntegration:
    def __init__(self, stellar_network: str, pi_network_config: dict):
        self.stellar_network = stellar_network
        self.pi_network_config = pi_network_config
        self.client = stellar_sdk.Client(stellar_network)

    def create_account(self, account_name: str, starting_balance: int) -> stellar_sdk.Account:
        # Create a new Stellar account with the specified name and starting balance
        account = self.client.create_account(account_name, starting_balance)
        return account

    def send_payment(self, source_account: stellar_sdk.Account, destination_account: stellar_sdk.Account, amount: int) -> stellar_sdk.Transaction:
        # Send a payment from the source account to the destination account
        transaction = self.client.send_payment(source_account, destination_account, amount)
        return transaction

    def get_account_balance(self, account: stellar_sdk.Account) -> int:
        # Get the current balance of the specified account
        balance = self.client.get_account_balance(account)
        return balance

# Example usage
pi_network_config = {
    "stellar_network": "testnet",
    "starting_balance": 1000
}

integration = PiNetworkStellarIntegration(pi_network_config["stellar_network"], pi_network_config)
account = integration.create_account("my_account", pi_network_config["starting_balance"])
print(account.account_id)

transaction = integration.send_payment(account, "destination_account", 500)
print(transaction.hash)

balance = integration.get_account_balance(account)
print(balance)
