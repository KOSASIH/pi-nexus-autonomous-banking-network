import mizuho


class MizuhoPaymentGateway:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        mizuho.Configuration.configure(api_key, api_secret)

    def create_payment(self, amount, currency):
        payment = mizuho.Payment.create({"amount": amount, "currency": currency})
        return payment
