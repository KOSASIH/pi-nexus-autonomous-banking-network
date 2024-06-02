import sumitomo_miyazaki


class SumitomoMiyazakiPaymentGateway:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        sumitomo_miyazaki.Configuration.configure(api_key, api_secret)

    def create_payment(self, amount, currency):
        payment = sumitomo_miyazaki.Payment.create(
            {"amount": amount, "currency": currency}
        )
        return payment
