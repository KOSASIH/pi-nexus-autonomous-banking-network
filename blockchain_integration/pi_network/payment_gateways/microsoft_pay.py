import microsoft_pay


class MicrosoftPayPaymentGateway:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        microsoft_pay.Configuration.configure(client_id, client_secret)

    def create_payment(self, amount, currency):
        payment = microsoft_pay.Payment.create({"amount": amount, "currency": currency})
        return payment
