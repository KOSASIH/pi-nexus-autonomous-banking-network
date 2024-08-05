class CurrencyExchangeRateManager:
    def __init__(self):
        self.exchange_rates = {}

    def set_exchange_rate(self, currency_pair, exchange_rate):
        self.exchange_rates[currency_pair] = exchange_rate

    def get_exchange_rate(self, currency_pair):
        return self.exchange_rates.get(currency_pair, 0)
