import wells_fargo


class WellsFargoPaymentGateway:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        wells_fargo.Configuration.configure(api_key, api_secret)

    def create_payment(self, amount, currency):
        payment = wells_fargo.Payment.create({"amount": amount, "currency": currency})
        return payment
