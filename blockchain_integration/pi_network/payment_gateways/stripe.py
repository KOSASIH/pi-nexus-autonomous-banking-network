import stripe


class StripePaymentGateway:
    def __init__(self, secret_key):
        self.secret_key = secret_key
        stripe.api_key = secret_key

    def create_charge(self, amount, currency, source):
        charge = stripe.Charge.create(
            amount=int(amount * 100), currency=currency, source=source
        )
        return charge
