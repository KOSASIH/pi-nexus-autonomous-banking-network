import visa_checkout

class VisaCheckoutPaymentGateway:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        visa_checkout.Configuration.configure(api_key, api_secret)

    def create_payment(self, amount, currency):
        payment = visa_checkout.Payment.create({
            'amount': amount,
            'currency': currency
        })
        return payment
