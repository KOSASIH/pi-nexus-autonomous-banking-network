import mastercard_payment_gateway


class MastercardPaymentGateway:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        mastercard_payment_gateway.Configuration.configure(api_key, api_secret)

    def create_payment(self, amount, currency):
        payment = mastercard_payment_gateway.Payment.create(
            {"amount": amount, "currency": currency}
        )
        return payment
