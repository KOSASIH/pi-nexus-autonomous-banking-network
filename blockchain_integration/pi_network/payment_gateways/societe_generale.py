import societe_generale

class SocieteGeneralePaymentGateway:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        societe_generale.Configuration.configure(api_key, api_secret)

    def create_payment(self, amount, currency):
        payment = societe_generale.Payment.create({
            'amount': amount,
            'currency': currency
        })
        return payment
