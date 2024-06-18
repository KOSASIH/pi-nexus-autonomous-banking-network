from stellar_utils import get_stellar_client, get_account_sequence, get_transaction_fee
from pi_network_api import PiNetworkAPI

class StellarPiBridge:
    def __init__(self, stellar_network: str, pi_network_api: PiNetworkAPI):
        self.stellar_network = stellar_network
        self.pi_network_api = pi_network_api
        self.stellar_client = get_stellar_client(stellar_network)

    def deposit_pi(self, user_id: str, amount: int) -> stellar_sdk.Transaction:
        """Deposit Pi into the user's Stellar account"""
        account_id = f"{user_id}-stellar"
        sequence = get_account_sequence(account_id, self.stellar_client)
        transaction = self.stellar_client.transaction_builder(
            source_account=account_id,
            sequence=sequence,
            operations=[
                stellar_sdk.Operation.payment(
                    destination=f"{user_id}-stellar",
                    asset_code="PI",
                    amount=str(amount)
                )
            ],
            fee=get_transaction_fee(self.stellar_client)
        )
        return transaction

    def withdraw_pi(self, user_id: str, amount: int) -> str:
        """Withdraw Pi from the user's Stellar account"""
        account_id = f"{user_id}-stellar"
        pi_balance = self.pi_network_api.get_user_balance(user_id)
        if pi_balance < amount:
            raise ValueError("Insufficient Pi balance")
        transaction_id = self.pi_network_api.send_pi_payment(user_id, "pi-reserve", amount)
        return transaction_id
