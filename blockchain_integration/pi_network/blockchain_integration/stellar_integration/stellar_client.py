import stellar_sdk

class StellarClient:
    def __init__(self, network_passphrase, horizon_url):
        self.network_passphrase = network_passphrase
        self.horizon_url = horizon_url
        self.server = stellar_sdk.Server(horizon_url)

    def get_account(self, account_id):
        return self.server.accounts().account_id(account_id).call()

    def create_account(self, keypair):
        transaction = stellar_sdk.TransactionBuilder(
            source_account=keypair,
            network_passphrase=self.network_passphrase,
            base_fee=100  # 0.01 XLM
        ).append_create_account_op(
            destination=keypair.public_key,
            starting_balance="10.0"  # 10 XLM
        ).build()
        transaction.sign(keypair)
        response = self.server.submit_transaction(transaction)
        return response

    def submit_transaction(self, transaction):
        return self.server.submit_transaction(transaction)
