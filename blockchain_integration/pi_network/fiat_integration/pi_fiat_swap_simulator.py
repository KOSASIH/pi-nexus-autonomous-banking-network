import random

class PiFiatSwapSimulator:
    def __init__(self, pi_network_api, fiat_exchange_rate_updater):
        self.pi_network_api = pi_network_api
        self.fiat_exchange_rate_updater = fiat_exchange_rate_updater

    def simulate_pi_fiat_swap(self, user_id, amount_pi, fiat_currency):
        pi_balance = self.pi_network_api.get_user_balance(user_id)
        fiat_exchange_rate = self.fiat_exchange_rate_updater.update_fiat_exchange_rate(fiat_currency)
        amount_fiat = pi_balance * fiat_exchange_rate
        # Simulate swap using random number generator
        simulated_swap = self.simulate_swap(amount_fiat)
        return simulated_swap
