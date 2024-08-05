import math

class PiFiatSwapOrchestrator:
    def __init__(self, pi_network_api, fiat_transaction_manager, bank_account_validator):
        self.pi_network_api = pi_network_api
        self.fiat_transaction_manager = fiat_transaction_manager
        self.bank_account_validator = bank_account_validator

    def swap_pi_for_fiat(self, user_id, amount_pi, fiat_currency):
        pi_balance = self.pi_network_api.get_user_balance(user_id)
        fiat_exchange_rate = self.fiat_transaction_manager.get_fiat_exchange_rate(fiat_currency)
        amount_fiat = pi_balance * fiat_exchange_rate
        bank_account_valid = self.bank_account_validator.validate_bank_account("1234567890", "John Doe")
        if bank_account_valid:
            self.fiat_transaction_manager.create_fiat_transaction(user_id, amount_fiat, fiat_currency)
            return True
        else:
            return False
