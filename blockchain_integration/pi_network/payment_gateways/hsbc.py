import hsbc

class HSBCPaymentGateway:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        hsbc.Configuration.configure(api_key, api_secret)

    def create_payment(self, amount, currency):
        payment = hsbc.Payment.create({
            'amount': amount,
            'currency': currency
        })
        return payment
