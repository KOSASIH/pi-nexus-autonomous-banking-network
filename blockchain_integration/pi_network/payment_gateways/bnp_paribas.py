import bnp_paribas

class BNPParibasPaymentGateway:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        bnp_paribas.Configuration.configure(api_key, api_secret)

    def create_payment(self, amount, currency):
        payment = bnp_paribas.Payment.create({
            'amount': amount,
            'currency': currency
        })
        return payment
