import dbs


class DBSPaymentGateway:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        dbs.Configuration.configure(api_key, api_secret)

    def create_payment(self, amount, currency):
        payment = dbs.Payment.create({"amount": amount, "currency": currency})
        return payment
