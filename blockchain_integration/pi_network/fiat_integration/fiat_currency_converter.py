import requests
import json

class FiatCurrencyConverter:
    def __init__(self, api_key):
        self.api_key = api_key
        self.exchange_rate_url = "https://api.exchangerate-api.com/v4/latest/"

    def get_exchange_rate(self, base_currency, target_currency):
        response = requests.get(self.exchange_rate_url + base_currency, params={"access_key": self.api_key})
        data = json.loads(response.text)
        if "rates" in data:
            return data["rates"][target_currency]
        else:
            raise ValueError("Failed to retrieve exchange rate")

    def convert_pi_to_fiat(self, pi_amount, fiat_currency):
        exchange_rate = self.get_exchange_rate("USD", fiat_currency)
        return pi_amount * exchange_rate

    def convert_fiat_to_pi(self, fiat_amount, fiat_currency):
        exchange_rate = self.get_exchange_rate(fiat_currency, "USD")
        return fiat_amount / exchange_rate

    def convert_pi_to_fiat_with_fee(self, pi_amount, fiat_currency, fee_percentage):
        fiat_amount = self.convert_pi_to_fiat(pi_amount, fiat_currency)
        fee = fiat_amount * (fee_percentage / 100)
        return fiat_amount - fee

    def convert_fiat_to_pi_with_fee(self, fiat_amount, fiat_currency, fee_percentage):
        pi_amount = self.convert_fiat_to_pi(fiat_amount, fiat_currency)
        fee = pi_amount * (fee_percentage / 100)
        return pi_amount - fee

def main():
    converter = FiatCurrencyConverter("YOUR_API_KEY")
    pi_amount = 100
    fiat_currency = "EUR"
    fiat_amount = converter.convert_pi_to_fiat(pi_amount, fiat_currency)
    print(f"{pi_amount} Pi = {fiat_amount} {fiat_currency}")
    fiat_amount = 1000
    pi_amount = converter.convert_fiat_to_pi(fiat_amount, fiat_currency)
    print(f"{fiat_amount} {fiat_currency} = {pi_amount} Pi")
    fiat_amount = converter.convert_pi_to_fiat_with_fee(pi_amount, fiat_currency, 2)
    print(f"{pi_amount} Pi = {fiat_amount} {fiat_currency} (with 2% fee)")
    pi_amount = converter.convert_fiat_to_pi_with_fee(fiat_amount, fiat_currency, 2)
    print(f"{fiat_amount} {fiat_currency} = {pi_amount} Pi (with 2% fee)")

if __name__ == "__main__":
    main()
