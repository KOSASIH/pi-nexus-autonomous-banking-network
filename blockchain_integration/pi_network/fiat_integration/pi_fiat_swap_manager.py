import math

class PiFiatSwapManager:
    def __init__(self, pi_network_api, fiat_exchange_rate_manager, bank_account_manager):
        self.pi_network_api = pi_network_api
        self.fiat_exchange_rate_manager = fiat_exchange_rate_manager
        self.bank_account_manager = bank_account_manager

    def swap_pi_for_fiat(self, user_id, amount_pi, fiat_currency):
        pi_balance = self.pi_network_api.get_user_balance(user_id)
        fiat_exchange_rate = self.fiat_exchange_rate_manager.get_fiat_exchange_rates()[fiat_currency]
        amount_fiat = pi_balance * fiat_exchange_rate
        bank_account_balance = self.bank_account_manager.get_bank_account_balance(user_id)
        if bank_account_balance >= amount_fiat:
            self.bank_account_manager.transfer_fiat_to_bank_account(user_id, amount_fiat)
            return True
        else:
            return False
