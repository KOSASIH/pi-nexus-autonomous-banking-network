import credit_suisse


class CreditSuissePaymentGateway:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        credit_suisse.Configuration.configure(api_key, api_secret)

    def create_payment(self, amount, currency):
        payment = credit_suisse.Payment.create({"amount": amount, "currency": currency})
        return payment
