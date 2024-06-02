import lloyds_bank


class LloydsBankPaymentGateway:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        lloyds_bank.Configuration.configure(api_key, api_secret)

    def create_payment(self, amount, currency):
        payment = lloyds_bank.Payment.create({"amount": amount, "currency": currency})
        return payment
