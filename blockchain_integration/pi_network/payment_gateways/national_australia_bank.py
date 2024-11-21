import national_australia_bank


class NationalAustraliaBankPaymentGateway:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        national_australia_bank.Configuration.configure(api_key, api_secret)

    def create_payment(self, amount, currency):
        payment = national_australia_bank.Payment.create(
            {"amount": amount, "currency": currency}
        )
        return payment
