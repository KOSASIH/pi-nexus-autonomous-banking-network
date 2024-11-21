import jpmorgan


class JPMorganPaymentGateway:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        jpmorgan.Configuration.configure(api_key, api_secret)

    def create_payment(self, amount, currency):
        payment = jpmorgan.Payment.create({"amount": amount, "currency": currency})
        return payment
