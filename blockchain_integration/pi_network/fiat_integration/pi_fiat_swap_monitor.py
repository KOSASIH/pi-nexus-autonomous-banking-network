import time

class PiFiatSwapMonitor:
    def __init__(self, pi_network_api, fiat_transaction_processor, bank_account_linker):
        self.pi_network_api = pi_network_api
        self.fiat_transaction_processor = fiat_transaction_processor
        self.bank_account_linker = bank_account_linker

    def monitor_pi_fiat_swap(self, user_id, amount_pi, fiat_currency):
        while True:
            pi_balance = self.pi_network_api.get_user_balance(user_id)
            if pi_balance >= amount_pi:
                fiat_exchange_rate = self.fiat_transaction_processor.get_fiat_exchange_rate(fiat_currency)
                amount_fiat = pi_balance * fiat_exchange_rate
                bank_account_linked = self.bank_account_linker.link_bank_account(user_id, "1234567890", "John Doe")
                if bank_account_linked:
                    self.fiat_transaction_processor.process_fiat_transaction(user_id, amount_fiat, fiat_currency)
                    break
            time.sleep(10)
