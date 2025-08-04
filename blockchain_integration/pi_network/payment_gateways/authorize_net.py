import authorize


class AuthorizeNetPaymentGateway:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        authorize.Configuration.configure(api_key, api_secret)

    def create_payment(self, amount, currency):
        payment = authorize.Payment()
        payment.amount = amount
        payment.currency = currency
        payment.create()
        return payment
