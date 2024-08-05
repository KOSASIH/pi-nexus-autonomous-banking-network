import math

class PiFiatExchangeRateCalculator:
    def __init__(self, pi_network_api, fiat_gateway):
        self.pi_network_api = pi_network_api
        self.fiat_gateway = fiat_gateway

    def calculate_exchange_rate(self, fiat_currency):
        pi_balance = self.pi_network_api.get_user_balance("user_id")
        fiat_rate = self.fiat_gateway.get_fiat_rates()[fiat_currency]
        exchange_rate = pi_balance / fiat_rate
        return exchange_rate
