class FiatWalletManager:
    def __init__(self):
        self.fiat_wallets = {}

    def create_fiat_wallet(self, user_id, fiat_currency):
        self.fiat_wallets[user_id] = fiat_currency

    def get_fiat_wallet(self, user_id):
        return self.fiat_wallets.get(user_id, {})
