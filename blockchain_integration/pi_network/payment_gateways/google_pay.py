import google_pay

class GooglePayPaymentGateway:
    def __init__(self, merchant_id, public_key):
        self.merchant_id = merchant_id
        self.public_key = public_key
        google_pay.Configuration.configure(merchant_id, public_key)

    def create_payment(self, amount, currency):
        payment = google_pay.Payment.create({
            'amount': amount,
            'currency': currency
        })
        return payment
