import web3


class BlockchainInterface:
    def __init__(self, endpoint):
        self.web3 = web3.Web3(web3.HTTPProvider(endpoint))

    def get_balance(self, account):
        return self.web3.eth.get_balance(account)

    def send_transaction(self, from_account, to_account, value):
        return self.web3.eth.send_transaction(
            {"from": from_account, "to": to_account, "value": value}
        )

    def get_transaction_receipt(self, transaction_hash):
        return self.web3.eth.get_transaction_receipt(transaction_hash)
