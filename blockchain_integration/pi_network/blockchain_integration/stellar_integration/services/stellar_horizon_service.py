from stellar_sdk.horizon import Horizon

class StellarHorizonService:
    def __init__(self, horizon_url):
        self.horizon_url = horizon_url
        self.horizon = Horizon(horizon_url)

    def get_account_info(self, public_key):
        return self.horizon.account_info(public_key)

    def get_transaction_history(self, public_key):
        return self.horizon.transactions(public_key)
