import math

class PiFiatSwapEngine:
    def __init__(self, pi_network_api, fiat_payment_gateway, bank_account_linker):
        self.pi_network_api = pi_network_api
        self.fiat_payment_gateway = fiat_payment_gateway
        self.bank_account_linker = bank_account_linker

    def swap_pi_for_fiat(self, user_id, amount_pi, fiat_currency):
        pi_balance = self.pi_network_api.get_user_balance(user_id)
        fiat_exchange_rate = self.fiat_payment_gateway.get_fiat_exchange_rate(fiat_currency)
        amount_fiat = pi_balance * fiat_exchange_rate
        bank_account_linked = self.bank_account_linker.link_bank_account(user_id, "1234567890", "John Doe")
        if bank_account_linked:
            self.fiat_payment_gateway.process_fiat_payment(user_id, amount_fiat, fiat_currency)
            return True
        else:
            return False
