import json
from web3 import Web3
from wallet_connector import WalletConnector

class MetaMaskAdapter:
    def __init__(self, wallet_connector, metamask_provider):
        self.wallet_connector = wallet_connector
        self.metamask_provider = metamask_provider

    def create_payment(self, recipient_address, amount, payment_identifier, user_uid):
        # Get the user's MetaMask account
        account = self.metamask_provider.request("eth_accounts")[0]

        # Create a payment using the wallet connector
        payment_response = self.wallet_connector.create_payment(recipient_address, amount, payment_identifier, user_uid)

        # Return the payment response
        return payment_response

    def complete_payment(self, payment_identifier, txid):
        # Complete the payment using the wallet connector
        complete_response = self.wallet_connector.complete_payment(payment_identifier, txid)

        # Return the complete response
        return complete_response
