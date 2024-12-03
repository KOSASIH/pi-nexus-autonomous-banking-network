import requests

class CurrencyConverter:
    def __init__(self):
        self.api_url = "https://api.exchangerate-api.com/v4/latest/USD"  # Example API

    def get_exchange_rates(self):
        response = requests.get(self.api_url)
        if response.status_code == 200:
            return response.json().get("rates", {})
        else:
            raise Exception("Failed to fetch exchange rates.")

    def convert(self, amount, from_currency, to_currency):
        rates = self.get_exchange_rates()
        if from_currency not in rates or to_currency not in rates:
            raise ValueError("Unsupported currency.")
        
        # Convert amount to USD first, then to the target currency
        amount_in_usd = amount / rates[from_currency]
        converted_amount = amount_in_usd * rates[to_currency]
        return converted_amount

    def get_exchange_rate(self, from_currency, to_currency):
        rates = self.get_exchange_rates()
        if from_currency not in rates or to_currency not in rates:
            raise ValueError("Unsupported currency.")
        
        return rates[to_currency] / rates[from_currency]
