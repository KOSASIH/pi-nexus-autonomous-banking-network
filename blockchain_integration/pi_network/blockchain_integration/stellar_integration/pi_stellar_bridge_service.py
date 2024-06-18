from stellar_account_manager import StellarAccountManager
from pi_network_api import PiNetworkAPI
from pi_stellar_bridge_config import PiStellarBridgeConfig

class PiStellarBridgeService:
    def __init__(self):
        self.config = PiStellarBridgeConfig()
        self.stellar_account_manager = StellarAccountManager(self.config.stellar_network)
        self.pi_network_api = PiNetworkAPI(self.config.pi_network_api_url, self.config.pi_network_api_key)

    def deposit_pi(self, user_id: str, amount: int) -> str:
        """Deposit Pi into the user's Stellar account"""
        account_id = self.stellar_account_manager.create_account(user_id)
        transaction = self.stellar_account_manager.fund_account(user_id, amount)
        return transaction.hash_hex()

    def withdraw_pi(self, user_id: str, amount: int) -> str:
        """Withdraw Pi from the user's Stellar account"""
        # TO DO: implement withdrawal logic
        pass
