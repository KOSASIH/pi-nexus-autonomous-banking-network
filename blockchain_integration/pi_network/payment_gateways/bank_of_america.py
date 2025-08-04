import bank_of_america


class BankOfAmericaPaymentGateway:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        bank_of_america.Configuration.configure(api_key, api_secret)

    def create_payment(self, amount, currency):
        payment = bank_of_america.Payment.create(
            {"amount": amount, "currency": currency}
        )
        return payment
