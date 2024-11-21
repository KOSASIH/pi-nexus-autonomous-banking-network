class PiWalletService:

    def __init__(self, config):
        self.config = config
        self.api_service = PiApiService(
            self.config["pi_network_api_endpoint"],
            self.config["pi_network_api_key"],
            self.config["pi_network_api_secret"],
        )
        self.blockchain_service = PiBlockchainService(self.api_service)

    def create_transaction(self, recipient, amount):
        sender = PiAccount(self.config["account_address"])
        sender.set_private_key(self.config["private_key"])
        sender.set_public_key(self.config["public_key"])
        transaction = PiTransaction(sender.get_public_key(), recipient, amount)
        transaction_dict = transaction.to_dict()
        transaction_dict["sender_private_key"] = sender.get_private_key()
        return transaction_dict

    def broadcast_transaction(self, transaction):
        return self.api_service.broadcast_transaction(transaction)

    def verify_transaction(self, transaction):
        return self.api_service.verify_transaction(transaction)
