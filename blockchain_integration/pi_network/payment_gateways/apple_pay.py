import apple_pay


class ApplePayPaymentGateway:
    def __init__(self, merchant_id, public_key):
        self.merchant_id = merchant_id
        self.public_key = public_key
        apple_pay.Configuration.configure(merchant_id, public_key)

    def create_payment(self, amount, currency):
        payment = apple_pay.Payment.create({"amount": amount, "currency": currency})
        return payment
